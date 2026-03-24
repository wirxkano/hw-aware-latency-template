import re
import glob


def parase_training_log(file_path):
    train_curve = []
    val_curve = []
    seen_iter = set()

    with open(file_path, "r") as f:
        for line in f:
            match = re.search(
                r"\[(\d+)\].*train-mae:(\d+\.\d+)\s*\|\s*val-mae:(\d+\.\d+)",
                line
            )
            if match:
                i = int(match.group(1))
                train = float(match.group(2))
                val = float(match.group(3))

                if i not in seen_iter:
                    train_curve.append(train)
                    val_curve.append(val)
                    seen_iter.add(i)

    return train_curve, val_curve


def parse_mae_mape_eval(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if "Evaluation" in line:
                mae = float(re.search(r"MAE:\s*([0-9.]+)", line).group(1))
                mape = float(re.search(r"MAPE:\s*([0-9.]+)%", line).group(1))
            
                return mae, mape
        
    return None, None

def parse_trade_off():
    results = []
    prefix = "experiments/xgboost/run_alpha"
    list_files_path = [f"{prefix}_0.log", f"{prefix}_0.05.log", f"{prefix}_0.1.log", f"{prefix}_0.2.log",
                       f"{prefix}_0.3.log", f"{prefix}_0.4.log", f"{prefix}_0.5.log"
                       ]
    for file in list_files_path:
        # extract alpha from filename
        alpha = float(re.search(r"alpha[_=]?([0-9]+(?:\.[0-9]+)?)", file).group(1))
        
        mae, mape = parse_mae_mape_eval(file)
        
        if mae is not None:
            results.append((alpha, mae, mape))

    # sort theo alpha
    results.sort(key=lambda x: x[0])

    alphas = [x[0] for x in results]
    maes   = [x[1] for x in results]
    mapes  = [x[2] for x in results]
    
    return results, alphas, maes, mapes
