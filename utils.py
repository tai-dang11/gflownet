import torch
import heapq
import random
import itertools
import numpy as np
from typing import List, Tuple, Union, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, QED, rdChemReactions
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig as PeftLoraConfig, get_peft_model
RDLogger.DisableLog('rdApp.*')

# =================================================================================================
# PART 1: CONFIGURATION
# =================================================================================================
class Config:
    MODEL_NAME = "merged_model/merged_sft_chemical_txgemma9b_epoch_20"
    BUILDING_BLOCKS_PATH = "data/short_building_blocks_subsampled_10000.txt"
    REACTIONS_PATH = "data/hb.txt"
    NUM_TRAINING_STEPS = 2000
    LEARNING_RATE = 5e-5
    LR_LOGZ = 1e-3
    BATCH_SIZE = 32
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    REPLAY_BUFFER_CAPACITY = 100
    MAX_TRAJECTORY_LENGTH = 10
    MAX_BUILDING_BLOCKS = 3
    MASK_BB_PATH = 'data/precomputed_bb_masks_enamine_bbs.npy'

    # Added from configs for the training function
    grad_acc = 1
    pf_temp_high = 2.0
    pf_temp_low = 0.5

    use_lora = True
    class LoraConfig:
        lora_rank = 16
        lora_alpha = 32
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_dropout = 0.1
        bias = "none"


# =================================================================================================
# PART 2: REPLAY BUFFER
# =================================================================================================
class ReplayBuffer:
    """
    A simple and clean implementation of a replay buffer for GFlowNet training.

    This buffer uses a min-heap to efficiently store a collection of the trajectories
    with the highest rewards. It supports prioritized sampling, where trajectories
    with higher rewards are more likely to be sampled.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer = []
        self.counter = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def has_items(self) -> bool:
        return len(self._buffer) > 0

    def add(self, trajectory: torch.Tensor, log_reward: float):
        """
        Adds a trajectory to the buffer.
        If the buffer is at capacity, it only adds the new trajectory if its
        reward is higher than the lowest reward currently in the buffer.
        """
        heap_item = (log_reward, self.counter, trajectory)
        if len(self._buffer) < self.capacity:
            heapq.heappush(self._buffer, heap_item)
        else:
            heapq.heappushpop(self._buffer, heap_item)
            
        self.counter += 1

    def add_batch(self, trajectories: List[torch.Tensor], log_rewards: torch.Tensor):
        """Adds a batch of trajectories and their corresponding log_rewards to the buffer."""
        for i in range(len(trajectories)):
            self.add(trajectories[i], log_rewards[i].item())

    def sample(self, batch_size: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Samples a batch of trajectories from the buffer using prioritized sampling.
        """
        if not self._buffer:
            return [], torch.tensor([])

        log_rewards = np.array([item[0] for item in self._buffer])
        rewards = np.exp(log_rewards - np.max(log_rewards)) # Stabilize softmax
        probabilities = rewards / np.sum(rewards)

        # Handle the case where all probabilities are zero
        if np.sum(probabilities) == 0:
            probabilities = np.ones(len(self._buffer)) / len(self._buffer)

        sampled_indices = np.random.choice(
            len(self._buffer),
            size=min(batch_size, len(self._buffer)), # Ensure we don't sample more than available
            p=probabilities,
            replace=True
        )
        
        ## --- CHANGE 4: Correctly unpack the 3-element tuple ---
        # We want the trajectory, which is now at index 2
        sampled_trajectories = [self._buffer[i][2] for i in sampled_indices] 
        # The log_reward is still at index 0
        sampled_log_rewards = torch.tensor([self._buffer[i][0] for i in sampled_indices], device=Config.DEVICE)

        return sampled_trajectories, sampled_log_rewards

class Reaction:
    """A helper class to wrap RDKit reactions with caching."""
    def __init__(self, template: str):
        self.template = template
        self.rxn = AllChem.ReactionFromSmarts(template)
        AllChem.SanitizeRxn(self.rxn)
        self.rxn.Initialize()
        self.num_reactants = self.rxn.GetNumReactantTemplates()
        self._is_reactant_cache = {}

    def is_reactant(self, mol: Chem.Mol) -> bool:
        smi = Chem.MolToSmiles(mol)
        if smi in self._is_reactant_cache:
            return self._is_reactant_cache[smi]
        result = self.rxn.IsMoleculeReactant(mol)
        self._is_reactant_cache[smi] = result
        return result

class ChemistryEngine:
    """Handles building-block addition, reaction execution, and template-based applicability with caching."""
    def __init__(self, building_block_path: str, reaction_path: str, precompute_bb_masks_path: str):
        print("Initializing Chemistry Engine...")
        # Load building blocks
        self.building_block_map = {}
        with open(building_block_path) as f:
            for i, line in enumerate(f):
                smi = line.strip()
                if not smi: continue
                self.building_block_map[f"<BB_{i}>"] = smi
        self.bb_tokens = list(self.building_block_map.keys())
        self.bb_mols = [Chem.MolFromSmiles(s) for s in self.building_block_map.values()]

        # Load reactions
        self.reactions = {}
        self.reactions_str = {}
        with open(reaction_path) as f:
            for i, line in enumerate(f):
                smarts = line.strip()
                if not smarts: continue
                self.reactions[f"<RXN_{i}>"] = Reaction(smarts)
                self.reactions_str[f"<RXN_{i}>"] = smarts
        self.rxn_tokens = list(self.reactions.keys())

        self.unimolecular_reactions = {t: r for t, r in self.reactions.items() if r.num_reactants == 1}
        self.bimolecular_reactions = {t: r for t, r in self.reactions.items() if r.num_reactants == 2}
        print(f"  - Loaded {len(self.unimolecular_reactions)} unimolecular and {len(self.bimolecular_reactions)} bimolecular reactions.")
        print(f"  - Loaded {len(self.building_block_map)} building blocks.")

        # Pre-computation for bimolecular reactions
        self._bb_compatibility_matrix = np.load(precompute_bb_masks_path)
        self._precomputed_any_bb_reacts = self._bb_compatibility_matrix.any(axis=2)
        print("  - Precomputed building block compatibility matrix.")
        print("Engine initialized successfully.\n")


    def execute_action(self, action_token: str, current_mols: list) -> Tuple[bool, list]:
        if action_token in self.building_block_map:
            smi = self.building_block_map[action_token]
            mol = Chem.MolFromSmiles(smi)
            return (True, current_mols + [mol]) if mol else (False, current_mols)

        if action_token in self.reactions:
            rxn = self.reactions[action_token].rxn
            n = rxn.GetNumReactantTemplates()
            if len(current_mols) < n:
                return False, current_mols

            reactants = tuple(current_mols[-n:])
            # For bimolecular, try both permutations
            reactant_perms = list(itertools.permutations(reactants, n)) if n > 1 else [reactants]
            for combo in reactant_perms:
                prods = rxn.RunReactants(combo)
                if len(prods) > 0:
                    for prod in prods:
                        try:
                            prod_canon = Chem.MolFromSmiles(Chem.MolToSmiles(prod[0], canonical=True))
                            if not prod_canon: continue
                            Chem.SanitizeMol(prod_canon)
                            prod_canon = Chem.RemoveHs(prod_canon)
                            return True, current_mols[:-n] + [prod_canon]
                        except Exception:
                            print(f'failed sanitize smile {Chem.MolToSmiles(prod_canon)}')
                            continue

        return False, current_mols

    def get_applicable_unimolecular(self, mol: Chem.Mol) -> List[str]:
        return [token for token, rxn in self.unimolecular_reactions.items() if rxn.is_reactant(mol)]

    def get_applicable_bimolecular(self, mol1: Chem.Mol) -> List[str]:
        valid_rxns = []
        bi_rxn_list = list(self.bimolecular_reactions.values())
        bi_rxn_tokens = list(self.bimolecular_reactions.keys())

        for i, reaction in enumerate(bi_rxn_list):
            reactant_templates = reaction.rxn.GetReactants()
            if mol1.HasSubstructMatch(reactant_templates[0]) and self._precomputed_any_bb_reacts[i, 1]:
                valid_rxns.append(bi_rxn_tokens[i])
            if mol1.HasSubstructMatch(reactant_templates[1]) and self._precomputed_any_bb_reacts[i, 0]:
                valid_rxns.append(bi_rxn_tokens[i])
        return list(set(valid_rxns)) # Return unique reactions

    def get_valid_partners(self, mol1: Chem.Mol, rxn_token: str) -> List[str]:
        if rxn_token not in self.bimolecular_reactions:
            return []

        reaction = self.bimolecular_reactions[rxn_token]
        bi_rxn_list = list(self.bimolecular_reactions.values())
        reaction_idx = bi_rxn_list.index(reaction)
        reactant_templates = reaction.rxn.GetReactants()
        valid_partners_indices = []

        if mol1.HasSubstructMatch(reactant_templates[0]):
            valid_partners_indices.extend(np.where(self._bb_compatibility_matrix[reaction_idx, 1, :])[0])
        if mol1.HasSubstructMatch(reactant_templates[1]):
            valid_partners_indices.extend(np.where(self._bb_compatibility_matrix[reaction_idx, 0, :])[0])
        return [self.bb_tokens[i] for i in set(valid_partners_indices)]

# =================================================================================================
# PART 3: PROMPTING AND BATCHED TRAJECTORY GENERATION
# =================================================================================================
def create_informative_prompt(
    workbench: list,
    trajectory_log: list,
    product_smi: list,
    chem_engine: ChemistryEngine
) -> str:
    """
    Creates a detailed prompt that now shows the intermediate product SMILES in the synthesis plan.
    """
    # Part 1: Current Workbench State (no change)
    wb_smis = [Chem.MolToSmiles(m) for m in workbench] if workbench else []
    wb_str = "\n".join(f"  - Mol {i}: {smi}" for i, smi in enumerate(wb_smis)) or "  - Empty"

    # --- Part 2: Synthesis Plan So Far (Correct Narrative Logic) ---
    plan_steps = []
    product_idx = 0
    step_counter = 1
    pending_bimolecular_rxn = None

    for token in trajectory_log:
        if token in chem_engine.building_block_map:
            smi = chem_engine.building_block_map[token]
            plan_steps.append(f"Step {step_counter}: Add {token}: ({smi})")
            step_counter += 1

            if pending_bimolecular_rxn:
                if product_idx < len(product_smi):
                    plan_steps.append(f"Step {step_counter}: Create Product -> {product_smi[product_idx]}")
                    step_counter += 1
                    product_idx += 1
                pending_bimolecular_rxn = None

        elif token in chem_engine.reactions:
            reaction = chem_engine.reactions[token]
            smarts = chem_engine.reactions_str[token]
            plan_steps.append(f"Step {step_counter}: Apply Reaction ({smarts})")
            step_counter += 1

            if reaction.num_reactants == 1:
                if product_idx < len(product_smi):
                    plan_steps.append(f"Step {step_counter}: Create Product -> {product_smi[product_idx]}")
                    step_counter += 1
                    product_idx += 1
            else:
                pending_bimolecular_rxn = True

    # Use a newline to join the steps into a multi-line block
    plan_str = "\n* ".join(plan_steps) if plan_steps else "None"
    if plan_steps:
        plan_str = "* " + plan_str

    # --- Final Assembled Prompt ---
    prompt = (
        f"You are a chemistry planning agent. Your goal is to synthesize a high-QED molecule.\n\n"
        f"## CURRENT STATE\n"
        f"* Workbench Contents:\n{wb_str}\n\n"
        f"## SYNTHESIS PLAN SO FAR\n{plan_str}\n\n"
        f"## TASK\n"
        f"Based on the current state, choose the best next action from the valid options below.\n"
        f"The format is: <BB_idx> or <RXN_idx> or <TERMINATE>\n\n "
        f"Your choice:\n"
    )
    
    return prompt

def reward_function(smiles_list: list) -> torch.Tensor:
    """
    Placeholder for your custom reward function.
    This should take a list of SMILES strings and return a tensor of reward values.
    """
    rewards = []
    for smiles in smiles_list:
        qed_score = qed_calculator(smiles)
        rewards.append(qed_score)
    return torch.tensor(rewards, dtype=torch.float32, device=Config.DEVICE)


def qed_calculator(smiles: str) -> float:
    """Calculates reward for a single SMILES string."""
    if not smiles: return 1e-8
    try:
        mol = Chem.MolFromSmiles(smiles)
        return QED.qed(mol) if mol else 1e-8
    except:
        return 1e-8

def setup_environment():
    print("--- Setting up environment ---")
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, torch_dtype=torch.bfloat16).to(Config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    if Config.use_lora:
        print("Applying LoRA to the model...")
        lora_config_params = Config.LoraConfig
        peft_config = PeftLoraConfig(
            r=lora_config_params.lora_rank,
            lora_alpha=lora_config_params.lora_alpha,
            target_modules=lora_config_params.target_modules,
            lora_dropout=lora_config_params.lora_dropout,
            bias=lora_config_params.bias,
        )
        model = get_peft_model(model, peft_config)
        print("LoRA applied successfully.")
        model.print_trainable_parameters()

    chem_engine = ChemistryEngine(Config.BUILDING_BLOCKS_PATH, Config.REACTIONS_PATH, Config.MASK_BB_PATH)
    special_tokens = (
        chem_engine.bb_tokens +
        chem_engine.rxn_tokens +
        ["<TERMINATE>"]
    )
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens, 'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    action_token_map = {tok: tokenizer.encode(tok, add_special_tokens=False)[0] for tok in special_tokens}
    print(f"Vocabulary size after adding special tokens: {len(tokenizer)}")
    print("--- Setup Complete ---")
    return model, tokenizer, chem_engine, action_token_map

