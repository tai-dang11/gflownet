import torch
import heapq
import random
import itertools
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional

from tqdm import tqdm
from rdkit import Chem, RDLogger
from utils import Config, ChemistryEngine, create_informative_prompt, setup_environment, ReplayBuffer, reward_function
RDLogger.DisableLog('rdApp.*')

def sample_llm(model, tokenizer, prompts, allowed_token_ids_list, temperature):
    if not prompts: return []
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(Config.DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        
    mask = torch.full_like(logits, float('-inf'))
    for i, allowed_ids in enumerate(allowed_token_ids_list):
        if allowed_ids: mask[i, allowed_ids] = 0

    filtered_logits = logits + mask
    log_probs = F.log_softmax(filtered_logits / temperature, dim=-1)
    probs = torch.exp(log_probs)
    chosen_token_ids = torch.multinomial(probs, num_samples=1)

    return chosen_token_ids.squeeze(-1), log_probs

def generate_trajectories(
    model, tokenizer, chem_engine: ChemistryEngine, action_token_map: dict,
    batch_size: int, temperature: float, action_seq: List[List[str]] = None
) -> Tuple[List, List, List, List]:
    # --- Initialization ---
    trajectory_logs = [[] for _ in range(batch_size)]
    trajectory_tensors = [[] for _ in range(batch_size)]
    workbenches = [[] for _ in range(batch_size)]
    products_smi = [[] for _ in range(batch_size)]
    pending_rxns = [None] * batch_size
    bb_counts = [0] * batch_size
    active_indices = list(range(batch_size))

    # Correctly initialize the results list to handle early termination
    completed_results = [None] * batch_size

    # Data structures for GFlowNet loss
    log_p_forwards = [[] for _ in range(batch_size)]
    log_p_eos = [[] for _ in range(batch_size)]

    # --- Step 0: Pick initial building block ---
    bb_tokens = chem_engine.bb_tokens
    allowed_ids = [action_token_map[t] for t in bb_tokens]
    initial_prompts = [create_informative_prompt([], [], [], chem_engine) for _ in active_indices]

    if initial_prompts:
        chosen_ids, log_probs = sample_llm(model, tokenizer, initial_prompts, [allowed_ids] * len(active_indices), temperature)

        indices_to_remove = []
        for i, original_index in enumerate(active_indices):
            chosen_id = chosen_ids[i]
            log_p_forwards[original_index].append(log_probs[i, chosen_id])
            log_p_eos[original_index].append(log_probs[i, action_token_map["<TERMINATE>"]])

            action_token = tokenizer.decode(chosen_id).strip()
            trajectory_logs[original_index].append(action_token)
            trajectory_tensors[original_index].append(chosen_id.clone().detach())

            succ, workbenches[original_index] = chem_engine.execute_action(action_token, workbenches[original_index])

            bb_counts[original_index] += 1

        if indices_to_remove:
            active_indices = [i for i in active_indices if i not in indices_to_remove]

    # === Main Generation Loop ===
    for step in range(1, Config.MAX_TRAJECTORY_LENGTH):
        if not active_indices: break

        indices_to_terminate_this_step = set()

        # --- Phase 1: Handle pending bimolecular reactions ---
        partner_phase_indices = [i for i in active_indices if pending_rxns[i]]

        if partner_phase_indices:
            prompts, allowed_ids_list, map_inf_to_batch = [], [], []
            for i in partner_phase_indices:
                partners = chem_engine.get_valid_partners(workbenches[i][-1], pending_rxns[i])
                prompts.append(create_informative_prompt(workbenches[i], trajectory_logs[i], products_smi[i], chem_engine))
                allowed_ids_list.append([action_token_map[t] for t in partners])
                map_inf_to_batch.append(i)

            if prompts:
                chosen_partner_ids, log_probs = sample_llm(model, tokenizer, prompts, allowed_ids_list, temperature)
                for j, i in enumerate(map_inf_to_batch):
                    chosen_id = chosen_partner_ids[j]
                    log_p_forwards[i].append(log_probs[j, chosen_id])
                    log_p_eos[i].append(log_probs[j, action_token_map["<TERMINATE>"]])

                    bb2 = tokenizer.decode(chosen_id).strip()
                    trajectory_logs[i].append(bb2)
                    trajectory_tensors[i].append(chosen_id.clone().detach())

                    succ_bb, tmp_wb = chem_engine.execute_action(bb2, workbenches[i])
                    if succ_bb: bb_counts[i] += 1

                    succ_rxn, workbenches[i] = chem_engine.execute_action(pending_rxns[i], tmp_wb)
                    products_smi[i].append(Chem.MolToSmiles(workbenches[i][-1]))
                    if not succ_rxn:
                        print(chem_engine.reactions_str[pending_rxns[i]])
                        print(f"\033[91mFailed to execute reaction {pending_rxns[i]}\033[0m")
                        indices_to_terminate_this_step.add(i)

                    pending_rxns[i] = None

        # --- Phase 2: Handle all other active trajectories ---
        action_phase_indices = [i for i in active_indices if i not in indices_to_terminate_this_step and not pending_rxns[i]]
        if not action_phase_indices:
            if indices_to_terminate_this_step:
                 active_indices = [i for i in active_indices if i not in indices_to_terminate_this_step]
            continue

        prompts, allowed_ids_list, map_inf_to_batch = [], [], []
        for i in action_phase_indices:
            possible_actions = ["<TERMINATE>"]
            if workbenches[i]:
                # possible_actions.extend(chem_engine.get_applicable_unimolecular(workbenches[i][-1]))

                # Only allow initiating new bimolecular reactions if we are under the building block limit.
                if bb_counts[i] < Config.MAX_BUILDING_BLOCKS:
                    possible_actions.extend(chem_engine.get_applicable_bimolecular(workbenches[i][-1]))
            # ----- END: CORRECTED ACTION SELECTION LOGIC -----

            # Get the token IDs for the allowed actions, removing duplicates
            allowed_ids = [action_token_map.get(t) for t in sorted(list(set(possible_actions))) if action_token_map.get(t)]

            if not allowed_ids: # If no actions are possible, terminate
                indices_to_terminate_this_step.add(i)
                continue

            prompts.append(create_informative_prompt(workbenches[i], trajectory_logs[i], products_smi[i], chem_engine))
            allowed_ids_list.append(allowed_ids)
            map_inf_to_batch.append(i)

        if prompts:
            chosen_action_ids, log_probs = sample_llm(model, tokenizer, prompts, allowed_ids_list, temperature)
            for j, i in enumerate(map_inf_to_batch):
                chosen_id = chosen_action_ids[j]
                log_p_forwards[i].append(log_probs[j, chosen_id])
                log_p_eos[i].append(log_probs[j, action_token_map["<TERMINATE>"]])

                chosen_token = tokenizer.decode(chosen_id).strip()
                trajectory_logs[i].append(chosen_token)
                trajectory_tensors[i].append(chosen_id.clone().detach())

                if chosen_token == "<TERMINATE>":
                    indices_to_terminate_this_step.add(i)
                elif chosen_token in chem_engine.unimolecular_reactions:
                    succ, workbenches[i] = chem_engine.execute_action(chosen_token, workbenches[i])
                    products_smi[i].append(Chem.MolToSmiles(workbenches[i][-1]))
                    if not succ:
                        indices_to_terminate_this_step.add(i)
                elif chosen_token in chem_engine.bimolecular_reactions:
                    pending_rxns[i] = chosen_token

        # --- Cleanup: Mark completed trajectories and update active list ---
        for i in list(indices_to_terminate_this_step):
            completed_results[i] = (trajectory_logs[i], workbenches[i])
        if indices_to_terminate_this_step:
            active_indices = [i for i in active_indices if i not in indices_to_terminate_this_step]

    # Handle trajectories that reached max length
    for i in active_indices:
        completed_results[i] = (trajectory_logs[i], workbenches[i])

    final_trajectory_tensors = [torch.stack(t) if t else torch.tensor([]) for t in trajectory_tensors]

    return completed_results, final_trajectory_tensors, log_p_forwards, log_p_eos

def train():
    """
    This function more accurately mimics the training algorithm found in main_infill.py,
    adapted for your ligand generation task.
    """
    config = Config()
    model, tokenizer, chem_engine, action_token_map = setup_environment()

    logZ = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float, device=config.DEVICE))
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': config.LEARNING_RATE},
        {'params': [logZ], 'lr': config.LR_LOGZ}
    ])

    rbuffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY)

    print("Starting GFlowNet Training")
    pb = tqdm(range(config.NUM_TRAINING_STEPS))

    all_losses = []
    all_rewards = []
    logging_window = 16 # The window size for the moving average

    for step in pb:
        optimizer.zero_grad()
        accumulated_loss = 0.0
        batch_rewards = []
        
        for _ in range(config.grad_acc):
            policy_choice = random.randint(0, 2)

            if policy_choice == 0:
                temperature = 1.0
                action_seq, logrewards_from_buffer = None, None
            elif policy_choice == 1 and rbuffer.has_items():
                temperature = 1.0
                action_seq, logrewards_from_buffer = rbuffer.sample(config.BATCH_SIZE)
            else:
                temperature = random.uniform(config.pf_temp_low, config.pf_temp_high)
                action_seq, logrewards_from_buffer = None, None

            completed, trajectory_tensors, log_p_forwards, log_p_eos = generate_trajectories(
                model, tokenizer, chem_engine, action_token_map, config.BATCH_SIZE, temperature
            )

            if logrewards_from_buffer is not None:
                log_rewards = logrewards_from_buffer
            else:
                final_smiles = [Chem.MolToSmiles(res[1][-1]) if res and res[1] else "" for res in completed]
                rewards = reward_function(final_smiles)

                for i, res in enumerate(completed):
                    if not res or not res[0] or res[0][-1] != "<TERMINATE>":
                        rewards[i] = 1e-8 # Assign a very small reward
                # --- END: ADDED PENALTY ---
                
                log_rewards = torch.log(rewards)

                if action_seq is None: # Only add if not from buffer
                    for i in range(len(trajectory_tensors)):
                        # We only add trajectories that resulted in a valid molecule
                        if completed[i] and completed[i][1]:
                            rbuffer.add(trajectory_tensors[i], log_rewards[i].item())

            batch_rewards.append(rewards.mean().item())

            # 3. CALCULATE TRAJECTORY BALANCE LOSS
            valid_indices = [
                i for i, res in enumerate(completed)
                if res and res[1] and log_p_forwards[i] and res[0][-1] == "<TERMINATE>"
            ]

            if not valid_indices:
                continue

            valid_log_p_forwards = [log_p_forwards[i] for i in valid_indices]
            valid_log_rewards = log_rewards[torch.tensor(valid_indices, device=config.DEVICE)]
            log_p_trajectory = torch.tensor([
                torch.stack(p).sum() for p in valid_log_p_forwards
            ], device=config.DEVICE)

            # Here is the core TB Loss calculation
            tb_loss = (logZ + log_p_trajectory - valid_log_rewards).pow(2).mean()

            if not torch.isfinite(tb_loss):
                print(f"Warning: Non-finite loss detected: {tb_loss.item()}. Skipping update.")
                continue
            
            # 4. BACKPROPAGATE LOSS (scaled for accumulation)
            loss_for_accumulation = tb_loss / config.grad_acc
            loss_for_accumulation.backward()
            accumulated_loss += loss_for_accumulation.item()

        
        optimizer.step()

        avg_loss_for_step = accumulated_loss / config.grad_acc if config.grad_acc > 0 else 0
        all_losses.append(avg_loss_for_step)
        all_rewards.append(np.mean(batch_rewards))
        
        #clean cuda memory
        torch.cuda.empty_cache()

        if step % logging_window == 0 and step > 0:
            start_index = max(0, len(all_losses) - logging_window)
            avg_loss = np.mean(all_losses[start_index:])
            avg_reward = np.mean(all_rewards[start_index:])

            print(f"Step {step}: Avg Loss (last {logging_window}) = {avg_loss:.4f}, Avg Reward (last {logging_window}) = {avg_reward:.4f}, LogZ = {logZ.item():.3f}")

if __name__ == "__main__":
    train()
