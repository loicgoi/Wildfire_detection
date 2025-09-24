# models/compare.py
from models.model import CONFIGS, load_model


def compare_models(configs, data_yaml="data.yaml"):
    results_summary = []

    for name, cfg in configs.items():
        print(f"\n=== Training {name} ({cfg['model']}) ===")
        model = load_model(cfg["model"])
        results = model.train(
            data=data_yaml, epochs=cfg["epochs"], batch=cfg["batch"], imgsz=640
        )
        results_summary.append(
            {
                "name": name,
                "model": cfg["model"],
                "batch": cfg["batch"],
                "epochs": cfg["epochs"],
                "metrics": results,
            }
        )

    return results_summary


if __name__ == "__main__":
    summary = compare_models(CONFIGS)
    print(summary)
