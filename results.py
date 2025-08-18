from setup import *
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def createConfusionMatrixDisplay(logger, predictions, datamodule):
    itemIdx = 0
    trueAnomalies = []
    predLabels = []
    if predictions is not None:
        for i, batch in enumerate(predictions):
            for j, prediction in enumerate(batch):
                trueAnomaly = datamodule.val_data.samples['label_index'][itemIdx]
                image_path = prediction.image_path
                anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
                predLabel = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
                trueAnomalies.append(trueAnomaly)
                predLabels.append(predLabel)
                itemIdx+=1
                pred_score = prediction.pred_score  # Image-level anomaly score
                
                pred = "Anomaly" if predLabel else "Normal"
                true = "Anomaly" if trueAnomaly else "Normal"
                logger.info(f"Predicted label: {pred}, True label: {true}, Anomaly score: {pred_score:1.2f}, Image path: {image_path}")
    trueAnomalies = np.asarray(trueAnomalies)
    predLabels = np.asarray(predLabels)
    confusionMatrix = confusion_matrix(trueAnomalies, predLabels)
    return confusionMatrix, trueAnomalies, predLabels
    
def logConfusionMatrix(logger, tblogger, confusionMatrix, trueAnomalies, predLabels):
    fig = plt.figure()
    ax = fig.subplots()
    
    CM_plot = ConfusionMatrixDisplay.from_predictions(trueAnomalies, predLabels, ax=ax)
    logger.info("Confusion Matrix:")
    logger.info(confusionMatrix)
    # CM_plot.figure_.savefig(os.path.join(prediction_path, f"{modelName}_confusion_matrix.png"))
    
    tblogger.add_image(CM_plot.figure_, "confusion_matrix", global_step=0)
    
def logMetrics(tblogger, res, confusionMatrix, takenTime, throughput):
    tp = confusionMatrix[1][1]
    tn = confusionMatrix[0][0]
    fp = confusionMatrix[0][1]
    fn = confusionMatrix[1][0]
    
    positive = tp + fn
    negative = tn + fp
    tpr = tp / positive
    tnr = tn / negative
    fnr = fn / positive
    fpr = fp / negative
    f1_score = 2 * tp/(2*tp + fp + fn)
    
    res[0]["image_positive"] = int(positive)
    res[0]["image_negative"] = int(negative)
    res[0]["image_tp"] = int(tp)
    res[0]["image_tn"] = int(tn)
    res[0]["image_fp"] = int(fp)
    res[0]["image_fn"] = int(fn)
    res[0]["image_TPR"] = float(tpr)
    res[0]["image_TNR"] = float(tnr)
    res[0]["image_FNR"] = float(fnr)
    res[0]["image_FPR"] = float(fpr)
    res[0]["taken_time"] = takenTime
    res[0]["throughput"] = throughput
    
    tblogger.log_metrics(metrics={"image_positive": positive,
                            "image_negative": negative,
                            "image_tp": tp,
                            "image_tn": tn,
                            "image_fp": fp,
                            "image_fn": fn,
                            "image_TPR": tpr,
                            "image_TNR": tnr,
                            "image_FNR": fnr,
                            "image_FPR": fpr,
                            "taken_time": takenTime,
                            "throughput": throughput},
                    step=0)