"""
Precompute species text embeddings from BioCLIP-2 for on-device lookup.

The iOS app only needs the vision encoder on-device. Text embeddings for
target species are precomputed here and shipped as a static lookup table.

At runtime:
    image → BioCLIP vision encoder → 768-dim vector
    cosine similarity with precomputed species embeddings
    → top-K species predictions → inject into Gemma prompt

Output:
    species_embeddings.npz  — {name: 768-dim vector} for each species
    species_list.json       — ordered list with metadata

Usage:
    # Default species list (hiking-relevant North American species)
    python src/precompute_embeddings.py --output_dir models/bioclip-mlx

    # Custom species list
    python src/precompute_embeddings.py \\
        --species_file my_species.txt \\
        --output_dir models/bioclip-mlx

    # With PlantNet species map
    python src/precompute_embeddings.py \\
        --plantnet_species_map plantnet300K_species_id_2_name.json \\
        --output_dir models/bioclip-mlx
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default species list — common in Eastern US hiking trails
# Matches the trails in hikeCompanion (Kildoo, Hells Hollow, Tranquil)
DEFAULT_SPECIES = [
    # --- Trees ---
    "Eastern Hemlock (Tsuga canadensis)",
    "White Oak (Quercus alba)",
    "Red Oak (Quercus rubra)",
    "Sugar Maple (Acer saccharum)",
    "Red Maple (Acer rubrum)",
    "American Beech (Fagus grandifolia)",
    "Tulip Poplar (Liriodendron tulipifera)",
    "Sassafras (Sassafras albidum)",
    "Yellow Birch (Betula alleghaniensis)",
    "American Elm (Ulmus americana)",
    "Black Cherry (Prunus serotina)",
    "Sycamore (Platanus occidentalis)",
    "White Pine (Pinus strobus)",
    "Virginia Pine (Pinus virginiana)",
    "Eastern Red Cedar (Juniperus virginiana)",
    "Black Walnut (Juglans nigra)",
    "Shagbark Hickory (Carya ovata)",
    "White Ash (Fraxinus americana)",
    "Flowering Dogwood (Cornus florida)",
    "American Chestnut (Castanea dentata)",
    "Black Locust (Robinia pseudoacacia)",
    "Honey Locust (Gleditsia triacanthos)",
    "Bald Cypress (Taxodium distichum)",
    # --- Shrubs ---
    "Rhododendron (Rhododendron maximum)",
    "Mountain Laurel (Kalmia latifolia)",
    "Witch Hazel (Hamamelis virginiana)",
    "Spicebush (Lindera benzoin)",
    "Blueberry (Vaccinium corymbosum)",
    "Elderberry (Sambucus nigra)",
    "Wild Hydrangea (Hydrangea arborescens)",
    # --- Wildflowers ---
    "Trillium (Trillium grandiflorum)",
    "Jack-in-the-Pulpit (Arisaema triphyllum)",
    "Virginia Bluebells (Mertensia virginica)",
    "Bloodroot (Sanguinaria canadensis)",
    "Wild Columbine (Aquilegia canadensis)",
    "Black-eyed Susan (Rudbera hirta)",
    "Cardinal Flower (Lobelia cardinalis)",
    "Goldenrod (Solidago canadensis)",
    "Purple Coneflower (Echinacea purpurea)",
    "Joe-Pye Weed (Eutrochium purpureum)",
    "Wild Ginger (Asarum canadense)",
    "Hepatica (Hepatica americana)",
    # --- Ferns & Mosses ---
    "Christmas Fern (Polystichum acrostichoides)",
    "Maidenhair Fern (Adiantum pedatum)",
    "Cinnamon Fern (Osmundastrum cinnamomeum)",
    "Hay-scented Fern (Dennstaedtia punctilobula)",
    "Sphagnum Moss (Sphagnum spp.)",
    "Haircap Moss (Polytrichum commune)",
    # --- Fungi ---
    "Turkey Tail (Trametes versicolor)",
    "Chicken of the Woods (Laetiporus sulphureus)",
    "Chanterelle (Cantharellus cibarius)",
    "Fly Agaric (Amanita muscaria)",
    "Reishi (Ganoderma lucidum)",
    "Morel (Morchella esculenta)",
    # --- Birds ---
    "Pileated Woodpecker (Dryocopus pileatus)",
    "Red-tailed Hawk (Buteo jamaicensis)",
    "Barred Owl (Strix varia)",
    "Wood Thrush (Hylocichla mustelina)",
    "Indigo Bunting (Passerina cyanea)",
    "Eastern Bluebird (Sialia sialis)",
    "American Robin (Turdus migratorius)",
    "Blue Jay (Cyanocitta cristata)",
    "Northern Cardinal (Cardinalis cardinalis)",
    "Downy Woodpecker (Dryobates pubescens)",
    "White-breasted Nuthatch (Sitta carolinensis)",
    "Ruby-throated Hummingbird (Archilochus colubris)",
    "Great Blue Heron (Ardea herodias)",
    "Bald Eagle (Haliaeetus leucocephalus)",
    "Wild Turkey (Meleagris gallopavo)",
    # --- Mammals ---
    "White-tailed Deer (Odocoileus virginianus)",
    "Gray Squirrel (Sciurus carolinensis)",
    "Red Fox (Vulpes vulpes)",
    "Eastern Chipmunk (Tamias striatus)",
    "Raccoon (Procyon lotor)",
    "Black Bear (Ursus americanus)",
    "Eastern Cottontail (Sylvilagus floridanus)",
    "Virginia Opossum (Didelphis virginiana)",
    "Striped Skunk (Mephitis mephitis)",
    "Groundhog (Marmota monax)",
    # --- Reptiles & Amphibians ---
    "Eastern Box Turtle (Terrapene carolina)",
    "Snapping Turtle (Chelydra serpentina)",
    "Green Frog (Lithobates clamitans)",
    "American Toad (Anaxyrus americanus)",
    "Red-spotted Newt (Notophthalmus viridescens)",
    "Northern Water Snake (Nerodia sipedon)",
    "Eastern Garter Snake (Thamnophis sirtalis)",
    # --- Insects ---
    "Monarch Butterfly (Danaus plexippus)",
    "Luna Moth (Actias luna)",
    "Firefly (Photinus pyralis)",
    "Dragonfly (Anisoptera)",
    "Eastern Tiger Swallowtail (Papilio glaucus)",
    # --- Rocks & Minerals (non-biological, but hikers ask) ---
    "sandstone",
    "limestone",
    "shale",
    "quartz",
    "mica schist",
    "granite",
    "iron oxide staining",
    "fossil",
    "lichen on rock",
    "moss on rock",
]

# Prompt templates for encoding species — BioCLIP works best with descriptive prompts
SPECIES_TEMPLATES = [
    "a photo of a {}",
    "a close-up photo of a {}",
    "a photo of a {} in the wild",
    "a {} in its natural habitat",
]


def load_bioclip(
    model_name: str = "hf-hub:imageomics/bioclip-2",
    bioclip_repo: str | None = None,
    device: str = "cpu",
):
    """Load BioCLIP-2 model and tokenizer."""
    if bioclip_repo:
        repo_src = Path(bioclip_repo) / "src"
        if repo_src.exists():
            sys.path.insert(0, str(repo_src))

    from open_clip import create_model_from_pretrained, get_tokenizer

    logger.info(f"Loading BioCLIP-2 from {model_name}...")
    model, preprocess = create_model_from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, tokenizer, preprocess


def encode_species_list(
    model, tokenizer, species_list: list[str], device: str = "cpu"
) -> dict[str, np.ndarray]:
    """
    Encode each species using averaged multi-template text embeddings.
    Returns {species_name: 768-dim normalized numpy vector}.
    """
    import torch

    embeddings = {}

    for species in species_list:
        # Generate text prompts from templates
        prompts = [t.format(species) for t in SPECIES_TEMPLATES]
        tokens = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(tokens, normalize=True)

        # Average across templates, re-normalize
        avg_feature = text_features.mean(dim=0)
        avg_feature = avg_feature / avg_feature.norm()

        embeddings[species] = avg_feature.cpu().numpy().astype(np.float16)

    return embeddings


def load_species_from_file(filepath: str) -> list[str]:
    """Load species list from a text file (one per line)."""
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def load_species_from_plantnet_map(filepath: str, max_species: int = 200) -> list[str]:
    """Load species names from PlantNet's species_id_2_name.json."""
    with open(filepath) as f:
        mapping = json.load(f)
    names = list(mapping.values())[:max_species]
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Precompute species text embeddings from BioCLIP-2"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--species_file", type=str, default=None, help="Custom species list (one per line)")
    parser.add_argument(
        "--plantnet_species_map",
        type=str,
        default=None,
        help="PlantNet species_id_2_name.json (adds PlantNet species to default list)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hf-hub:imageomics/bioclip-2",
        help="BioCLIP model name",
    )
    parser.add_argument("--bioclip_repo", type=str, default=None, help="Local bioclip-2 repo path")
    parser.add_argument("--device", type=str, default="cpu", help="Device for text encoding")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build species list
    if args.species_file:
        species_list = load_species_from_file(args.species_file)
        logger.info(f"Loaded {len(species_list)} species from {args.species_file}")
    else:
        species_list = list(DEFAULT_SPECIES)
        logger.info(f"Using default species list ({len(species_list)} species)")

    if args.plantnet_species_map:
        pn_species = load_species_from_plantnet_map(args.plantnet_species_map)
        # Add PlantNet species not already in list
        existing = set(s.lower() for s in species_list)
        added = 0
        for s in pn_species:
            if s.lower() not in existing:
                species_list.append(s)
                existing.add(s.lower())
                added += 1
        logger.info(f"Added {added} species from PlantNet map (total: {len(species_list)})")

    # Load model and encode
    model, tokenizer, _ = load_bioclip(args.model_name, args.bioclip_repo, args.device)

    logger.info(f"Encoding {len(species_list)} species...")
    embeddings = encode_species_list(model, tokenizer, species_list, args.device)

    # Save as npz (compact binary format)
    emb_path = output_dir / "species_embeddings.npz"
    np.savez_compressed(str(emb_path), **{k: v for k, v in embeddings.items()})
    emb_size = emb_path.stat().st_size
    logger.info(f"Saved embeddings: {emb_path} ({emb_size / 1e3:.1f} KB)")

    # Also save as a structured JSON for the iOS app to parse species metadata
    species_meta = []
    for i, name in enumerate(species_list):
        # Parse common name and scientific name if format is "Common (Scientific)"
        common = name
        scientific = None
        if "(" in name and name.endswith(")"):
            common = name[: name.index("(")].strip()
            scientific = name[name.index("(") + 1 : -1].strip()

        species_meta.append(
            {
                "index": i,
                "name": name,
                "common_name": common,
                "scientific_name": scientific,
            }
        )

    meta_path = output_dir / "species_list.json"
    with open(meta_path, "w") as f:
        json.dump(species_meta, f, indent=2)
    logger.info(f"Saved species metadata: {meta_path}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Species: {len(species_list)}")
    logger.info(f"Embedding dim: 768 (float16)")
    logger.info(f"Embeddings file: {emb_size / 1e3:.1f} KB")
    logger.info("")
    logger.info("Files to copy to iOS bundle:")
    logger.info(f"  {emb_path}  → Models/BioCLIP/species_embeddings.npz")
    logger.info(f"  {meta_path} → Models/BioCLIP/species_list.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
