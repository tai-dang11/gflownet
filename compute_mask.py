import warnings
import argparse
from typing import List

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# Suppress RDKit warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
RDLogger.DisableLog('rdApp.*')



class Reaction:
    """A helper class to wrap RDKit reactions, used by the pre-computation logic."""

    def __init__(self, template: str):
        self.template = template
        self.rxn = AllChem.ReactionFromSmarts(template)
        AllChem.SanitizeRxn(self.rxn)
        self.num_reactants = self.rxn.GetNumReactantTemplates()


def precompute_bb_masks(
    bimolecular_reactions: List[Reaction], building_blocks_mols: List[Chem.Mol]
) -> np.ndarray:
    """
    Creates a lookup matrix for bimolecular reactions.

    Args:
        bimolecular_reactions (List[Reaction]): A list of bimolecular Reaction objects.
        building_blocks_mols (List[Chem.Mol]): A list of building block RDKit Mol objects.

    Returns:
        np.ndarray: A boolean matrix of shape (num_bimolecular_reactions, 2, num_building_blocks).
                    Value [i, j, k] is True if building block k is a valid reactant for
                    reactant slot j of reaction i.
    """
    num_bi_rxns = len(bimolecular_reactions)
    num_bbs = len(building_blocks_mols)

    # The matrix holds [reaction_idx, reactant_slot_idx, building_block_idx]
    compatibility_matrix = np.zeros((num_bi_rxns, 2, num_bbs), dtype=bool)

    print(f"Processing {num_bi_rxns} reactions and {num_bbs} building blocks...")

    for i, reaction in enumerate(bimolecular_reactions):
        if (i + 1) % 10 == 0:
            print(f"  ...processed {i+1}/{num_bi_rxns} reactions...")

        reactant_templates = reaction.rxn.GetReactants()
        for k, bb_mol in enumerate(building_blocks_mols):
            if bb_mol is None:
                continue
            # Check if the building block matches the first reactant template
            if bb_mol.HasSubstructMatch(reactant_templates[0]):
                compatibility_matrix[i, 0, k] = True
            # Check if the building block matches the second reactant template
            if bb_mol.HasSubstructMatch(reactant_templates[1]):
                compatibility_matrix[i, 1, k] = True

    print("Finished pre-computation.")
    return compatibility_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute a compatibility mask for bimolecular reactions and building blocks."
    )
    parser.add_argument(
        "--reactions", type=str, default="data/hb.txt", help="Path to the reaction templates file."
    )
    parser.add_argument(
        "--blocks", type=str, default="data/short_building_blocks_subsampled_10000.txt", help="Path to the building blocks file."
    )
    parser.add_argument(
        "--output", type=str, default="data/precomputed_bb_masks_enamine_bbs.npy", help="Path to save the output .npy file."
    )
    args = parser.parse_args()

    # --- Step 1: Create dummy data files for the example ---

    # --- Step 2: Load data from files ---
    print("\nLoading data from files...")
    with open(args.reactions, 'r') as f:
        all_templates = [line.strip() for line in f if line.strip()]
    with open(args.blocks, 'r') as f:
        bb_smiles = [line.strip() for line in f if line.strip()]

    # Convert to RDKit objects
    all_reactions = [Reaction(t) for t in all_templates]
    bimolecular_reactions = [r for r in all_reactions if r.num_reactants == 2]
    building_blocks_mols = [Chem.MolFromSmiles(s) for s in bb_smiles]

    # --- Step 3: Run the pre-computation ---
    print("\nStarting pre-computation...")
    bb_mask = precompute_bb_masks(bimolecular_reactions, building_blocks_mols)

    # --- Step 4: Save the result ---
    np.save(args.output, bb_mask)
    print(f"\nCompatibility matrix saved to '{args.output}'")
    print(f"Matrix shape: {bb_mask.shape}")
    print("\nExample: To load this mask later, use `mask = np.load('precomputed_mask.npy')`")