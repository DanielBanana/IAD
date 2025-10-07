import argparse
from anomalib.data import MVTecAD, BTech, Visa, Kolektor
from anomalib.models import Padim, Patchcore, EfficientAd, Dsr, ReverseDistillation, Fastflow, Stfpm
from anomalib.pipelines.benchmark.job import BenchmarkJob

# Define available models, datasets, and categories
MODELS = ["efficientad-s", "efficientad-m", "patchcore", "padim", "dsr", "reverse_distillation", "rd", "stfpm", "fastflow"]
DATASETS = ["mvtecad", "kolektor", "visa", "btech"]
CATEGORIES = {
    "mvtecad": ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper", "carpet", "grid", "leather", "tile", "wood"],
    "kolektor": ["none"],
    "visa": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
    "btech": ["01", "02", "03"]
}

def main(model_name, dataset, category):
    # Initialize model
    if model_name == "efficientad-s":
        model = EfficientAd(model_size='small')
    elif model_name == "efficientad-m":
        model = EfficientAd(model_size='medium')
    elif model_name == "patchcore":
        model = Patchcore()
    elif model_name == "padim":
        model = Padim()
    elif model_name == "dsr":
        model = Dsr()
    elif model_name == "reverse_distillation":
        model = ReverseDistillation()
    elif model_name == "rd":
        model = ReverseDistillation()
    elif model_name == "stfpm":
        model = Stfpm()
    elif model_name == "fastflow":
        model = Fastflow()
    else:
        raise ValueError(f"Model {model_name} not found! Available models are: {', '.join(MODELS)}")

    # Initialize datamodule
    if dataset == "mvtecad":
        datamodule = MVTecAD(category=category, train_batch_size=1, eval_batch_size=1)
    elif dataset == "kolektor":
        datamodule = Kolektor(train_batch_size=1, eval_batch_size=1)
    elif dataset == "visa":
        datamodule = Visa(category=category, train_batch_size=1, eval_batch_size=1)
    elif dataset == "btech":
        datamodule = BTech(category=category, train_batch_size=1, eval_batch_size=1)
    else:
        raise ValueError(f"Dataset {dataset} not found! Available datasets are: {', '.join(DATASETS)}")

    # Initialize and run the benchmark job
    job = BenchmarkJob(
        accelerator="cpu",
        model=model,
        datamodule=datamodule,
        seed=42,
        flat_cfg={
            "model": model_name,
            "dataset": dataset,
            "dataset.category": category,
            "dataset.train_batch_size": 1,
            "dataset.eval_batch_size": 1
        }
    )

    # Run the benchmark job
    results = job.run()
    coll_results = job.collect([results])
    job.save(coll_results)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Benchmark an anomaly detection model.")
    parser.add_argument("--modelName", type=str, default="efficientad-s", help="Which model to benchmark")
    parser.add_argument("--dataset", type=str, default="mvtecad", help="Which dataset to benchmark on")
    parser.add_argument("--category", type=str, default="cable", help="Which category to benchmark on")

    # Parse the arguments
    args = parser.parse_args()

    model_name = args.model.lower()
    dataset = args.dataset.lower()
    category = args.category.lower()

    if model_name not in MODELS:
        print(f"Model {model_name} not found! Available models are: {', '.join(MODELS)}")
        exit(1)

    if dataset not in DATASETS:
        print(f"Dataset {dataset} not found! Available datasets are: {', '.join(DATASETS)}")
        exit(1)

    if category not in CATEGORIES[dataset]:
        print(f"Category {category} not found! Available categories are: {', '.join(CATEGORIES[dataset])}")
        exit(1)

    # Call the main function with parsed arguments
    main(model_name, dataset, category)
