import cv2
from pipeline import AnimalAnalysisPipeline, CLASS_NAMES, DEVICE

pipeline = AnimalAnalysisPipeline(
    yolo_weights="yolov8n.pt",
    classifier_weights="weights/efficientnet_b0_best.pth",
    num_classes=len(CLASS_NAMES),
    class_names=CLASS_NAMES,
)

cap = cv2.VideoCapture(0)  # 0 = laptop webcam
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = pipeline.run(frame)

    annotated = cv2.cvtColor(result["annotated_rgb"], cv2.COLOR_RGB2BGR)
    label = f"{result['predicted_class'].upper()} {result['confidence']:.1%}"
    cv2.putText(annotated, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f"{result['total_ms']:.0f}ms", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Animal Analysis - Live", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()