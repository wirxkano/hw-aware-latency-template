import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.parse_log import parase_training_log, parse_trade_off

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

def draw_learning_curve():
    train_curve, val_curve = parase_training_log("experiments/xgboost/run_20260324_155205.log")
    plt.figure(figsize=(12, 6))

    plt.plot(train_curve, label="Train MAE")
    plt.plot(val_curve, label="Validation MAE")

    plt.xlabel("Iteration")
    plt.ylabel("MAE")
    plt.title("Learning Curve")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("figs/learning_curve.pdf", dpi=300)
    
    
def draw_trade_off():
    results, alphas, maes, mapes = parse_trade_off()
    plt.figure(figsize=(12, 8))
    plt.plot(maes, mapes, marker='o', linewidth=3)

    for a, mae, mape in results:
        # plt.text(mae, mape, fontsize=14)
        plt.annotate(f"{a}", 
             xy=(mae, mape), 
             xytext=(0, 6), 
             fontsize=14,
             fontweight="bold",
             textcoords="offset points")

    plt.xlabel("MAE")
    plt.ylabel("MAPE (%)")
    plt.title("Trade-off between MAE and MAPE")

    plt.grid()
    plt.tight_layout()
    plt.savefig("figs/mae_mape_tradeoff.pdf", dpi=300)
    
    # fig, ax1 = plt.subplots()

    # ax1.plot(alphas, maes, marker='o')
    # ax1.set_xlabel("alpha")
    # ax1.set_ylabel("MAE")

    # ax2 = ax1.twinx()
    # ax2.plot(alphas, mapes, linestyle='--', marker='s')
    # ax2.set_ylabel("MAPE (%)")

    # plt.title("Effect of alpha on MAE and MAPE")
    # plt.tight_layout()
    # plt.savefig("figs/mae_mape_tradeoff.pdf", dpi=300)
    
    
    
if __name__ == "__main__":
    draw_trade_off()