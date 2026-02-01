using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.InferenceEngine;
using UnityEngine.Android;
using System.Linq;

public class QuestYolo : MonoBehaviour
{
    [Header("Debug")]
    public TMP_Text debugText;

    [Header("Assets")]
    public ModelAsset modelAsset;
    public RectTransform canvasRect;
    public GameObject boxPrefab;
    public RawImage previewImage;

    // ===== YOLOv8 =====
    private const int INPUT_SIZE = 640;
    private const int NUM_ANCHORS = 8400;
    private const int NUM_CLASSES = 80;
    private const int CHANNELS = 4 + NUM_CLASSES; // 84

    private readonly string[] classNames =
    {
        "Person","Bicycle","Car","Motorcycle","Airplane","Bus","Train","Truck","Boat","Traffic Light",
        "Fire Hydrant","Stop Sign","Parking Meter","Bench","Bird","Cat","Dog","Horse","Sheep","Cow",
        "Elephant","Bear","Zebra","Giraffe","Backpack","Umbrella","Handbag","Tie","Suitcase","Frisbee",
        "Skis","Snowboard","Sports Ball","Kite","Baseball Bat","Baseball Glove","Skateboard","Surfboard",
        "Tennis Racket","Bottle","Wine Glass","Cup","Fork","Knife","Spoon","Bowl","Banana","Apple",
        "Sandwich","Orange","Broccoli","Carrot","Hot Dog","Pizza","Donut","Cake","Chair","Couch",
        "Potted Plant","Bed","Dining Table","Toilet","TV","Laptop","Mouse","Remote","Keyboard","Cell Phone",
        "Microwave","Oven","Toaster","Sink","Refrigerator","Book","Clock","Vase","Scissors","Teddy Bear",
        "Hair Drier","Toothbrush"
    };

    // 只偵測人與交通工具
    private readonly HashSet<int> targetIds = new() { 0, 1, 2, 3, 5, 7 };

    private WebCamTexture webcam;
    private Worker worker;
    private Tensor<float> inputTensor;

    private readonly List<GameObject> activeBoxes = new();

    void Start()
    {
        if (previewImage == null)
            previewImage = GetComponentInChildren<RawImage>();

        Log("Init YOLOv8 (Sentis / CPU)");
        StartCoroutine(InitCameraAndAI());
    }

    void Log(string msg)
    {
        Debug.Log(msg);
        if (debugText != null)
            debugText.text = msg;
    }

    IEnumerator InitCameraAndAI()
    {
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
            yield return new WaitForSeconds(1f);
        }

        var devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Log("No camera found");
            yield break;
        }

        webcam = new WebCamTexture(devices[0].name, 1280, 720, 30);
        webcam.Play();

        while (webcam.width <= 16)
            yield return null;

        previewImage.texture = webcam;
        previewImage.color = Color.white;

        Log("Loading model...");
        Model model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.CPU);

        Log("AI Ready");
        StartCoroutine(AIUpdateLoop());
    }

    IEnumerator AIUpdateLoop()
    {
        int frame = 0;
        while (true)
        {
            frame++;
            if (webcam.didUpdateThisFrame)
                RunAI(frame);
            yield return null;
        }
    }

    void RunAI(int frame)
    {
        inputTensor?.Dispose();
        inputTensor = TextureConverter.ToTensor(webcam, INPUT_SIZE, INPUT_SIZE, 3);
        worker.Schedule(inputTensor);

        Tensor output = worker.PeekOutput();
        if (output is Tensor<float> t)
        {
            float[] data = t.DownloadToArray();
            ParseYoloV8Output(data, frame);
        }
    }

    // =========================================================
    // YOLOv8 output: [1, 84, 8400]
    // =========================================================
    void ParseYoloV8Output(float[] data, int frame)
    {
        List<Detection> detections = new();
        float maxScore = 0f;

        for (int i = 0; i < NUM_ANCHORS; i++)
        {
            int baseIdx = i * CHANNELS;

            float cx = data[baseIdx + 0];
            float cy = data[baseIdx + 1];
            float w = data[baseIdx + 2];
            float h = data[baseIdx + 3];

            float bestScore = 0f;
            int bestClass = -1;

            for (int c = 0; c < NUM_CLASSES; c++)
            {
                float score = data[baseIdx + 4 + c];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            maxScore = Mathf.Max(maxScore, bestScore);

            if (bestScore > 0.25f && targetIds.Contains(bestClass))
            {
                detections.Add(new Detection
                {
                    classId = bestClass,
                    score = bestScore,
                    // ?? 不假設 pixel，直接存 raw
                    box = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h)
                });
            }
        }

        DrawBoxes(NMS(detections), maxScore, frame);
    }

    // =========================================================

    void DrawBoxes(List<Detection> dets, float maxScore, int frame)
    {
        if (debugText != null)
        {
            debugText.text =
                $"Frame: {frame}\n" +
                $"Objects: {dets.Count}\n" +
                $"MaxScore: {maxScore:F2}";
        }

        foreach (var b in activeBoxes)
            Destroy(b);
        activeBoxes.Clear();

        float cw = canvasRect.rect.width;
        float ch = canvasRect.rect.height;

        foreach (var d in dets)
        {
            GameObject box = Instantiate(boxPrefab, canvasRect);
            activeBoxes.Add(box);

            RectTransform rt = box.GetComponent<RectTransform>();
            rt.pivot = Vector2.zero;
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.zero;

            // ===============================
            // ?? 關鍵修正：當作 normalized 0~1
            // ===============================
            float x = Mathf.Clamp01(d.box.x);
            float y = Mathf.Clamp01(d.box.y);
            float w = Mathf.Clamp01(d.box.width);
            float h = Mathf.Clamp01(d.box.height);

            rt.anchoredPosition = new Vector2(
                x * cw,
                (1f - (y + h)) * ch
            );

            rt.sizeDelta = new Vector2(
                w * cw,
                h * ch
            );

            TMP_Text t = box.GetComponentInChildren<TMP_Text>();
            if (t != null)
                t.text = $"{classNames[d.classId]} {d.score:F2}";
        }
    }

    // ===== NMS =====
    List<Detection> NMS(List<Detection> dets)
    {
        List<Detection> result = new();
        var sorted = dets.OrderByDescending(d => d.score).ToList();

        while (sorted.Count > 0)
        {
            var a = sorted[0];
            result.Add(a);
            sorted.RemoveAt(0);
            sorted.RemoveAll(b => IoU(a.box, b.box) > 0.5f);
        }
        return result;
    }

    float IoU(Rect a, Rect b)
    {
        float x1 = Mathf.Max(a.x, b.x);
        float y1 = Mathf.Max(a.y, b.y);
        float x2 = Mathf.Min(a.x + a.width, b.x + b.width);
        float y2 = Mathf.Min(a.y + a.height, b.y + b.height);
        float w = Mathf.Max(0, x2 - x1);
        float h = Mathf.Max(0, y2 - y1);
        return (w * h) / (a.width * a.height + b.width * b.height - w * h);
    }

    struct Detection
    {
        public int classId;
        public float score;
        public Rect box;
    }

    void OnDestroy()
    {
        worker?.Dispose();
        inputTensor?.Dispose();
        if (webcam != null) webcam.Stop();
    }
}
