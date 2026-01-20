using UnityEngine;
using UnityEngine.UI;

using System.Collections.Generic;
using TMPro; // [新增] 引用 TextMeshPro

public class QuestYolo : MonoBehaviour
{
    [Header("檔案設定")]
    public Unity.InferenceEngine.ModelAsset modelAsset;
    public ComputeShader postProcess;
    public RectTransform canvasRect;
    public GameObject boxPrefab;

    // [新增] YOLO 的 80 個類別名稱 (COCO Dataset)
    private readonly string[] classNames = new string[] {
        "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat", "Traffic Light",
        "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow",
        "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee",
        "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat", "Baseball Glove", "Skateboard", "Surfboard",
        "Tennis Racket", "Bottle", "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
        "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Chair", "Couch",
        "Potted Plant", "Bed", "Dining Table", "Toilet", "TV", "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone",
        "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy Bear",
        "Hair Drier", "Toothbrush"
    };

    private WebCamTexture webcam;
    private Unity.InferenceEngine.Worker worker;
    private ComputeBuffer boxBuf, classBuf, scoreBuf, countBuf;
    private List<GameObject> activeBoxes = new List<GameObject>();
    private Unity.InferenceEngine.Tensor<float> inputTensor;

    void Start()
    {
        // 為了效能與對焦，建議明確指定鏡頭 (通常 Quest 外鏡頭不是預設 0 號)
        // 這裡維持簡單，若畫面全黑我們再來調整
        webcam = new WebCamTexture(1280, 720, 30);
        webcam.Play();

        Unity.InferenceEngine.Model model = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
        worker = new Unity.InferenceEngine.Worker(model, Unity.InferenceEngine.BackendType.GPUCompute);

        boxBuf = new ComputeBuffer(8400, sizeof(float) * 4);
        classBuf = new ComputeBuffer(8400, sizeof(int));
        scoreBuf = new ComputeBuffer(8400, sizeof(float));
        countBuf = new ComputeBuffer(1, sizeof(uint));
    }

    void Update()
    {
        if (webcam.didUpdateThisFrame) RunAI();
    }

    void RunAI()
    {
        if (inputTensor != null) inputTensor.Dispose();
        inputTensor = Unity.InferenceEngine.TextureConverter.ToTensor(webcam, 640, 640, 3);
        worker.Schedule(inputTensor);
        Unity.InferenceEngine.Tensor<float> output = worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;
        Decode(output);
    }

    void Decode(Unity.InferenceEngine.Tensor<float> output)
    {
        countBuf.SetData(new uint[] { 0 });
        var computeData = Unity.InferenceEngine.ComputeTensorData.Pin(output);

        int k = postProcess.FindKernel("CSMain");
        postProcess.SetBuffer(k, "outputTensor", computeData.buffer);
        postProcess.SetBuffer(k, "boxes", boxBuf);
        postProcess.SetBuffer(k, "classes", classBuf);
        postProcess.SetBuffer(k, "scores", scoreBuf);
        postProcess.SetBuffer(k, "counter", countBuf);

        postProcess.SetFloat("confidenceThreshold", 0.5f);
        postProcess.SetInt("numDetections", 8400);
        postProcess.SetInt("numClasses", 80);

        postProcess.Dispatch(k, Mathf.CeilToInt(8400 / 256f), 1, 1);

        uint[] countArr = new uint[1];
        countBuf.GetData(countArr);
        int count = (int)countArr[0];

        Vector4[] boxes = new Vector4[count];
        int[] classes = new int[count]; // [新增] 需要讀取類別 ID

        if (count > 0)
        {
            boxBuf.GetData(boxes, 0, 0, count);
            classBuf.GetData(classes, 0, 0, count); // [新增] 從 GPU 讀回類別
        }

        computeData.Dispose();
        DrawBoxes(boxes, classes, count); // [修改] 多傳入 classes
    }

    void DrawBoxes(Vector4[] boxes, int[] classes, int count)
    {
        foreach (var b in activeBoxes) Destroy(b);
        activeBoxes.Clear();

        float w = canvasRect.rect.width;
        float h = canvasRect.rect.height;

        for (int i = 0; i < count; i++)
        {
            GameObject box = Instantiate(boxPrefab, canvasRect);
            activeBoxes.Add(box);
            RectTransform rt = box.GetComponent<RectTransform>();

            float nX = boxes[i].x / 640f;
            float nY = boxes[i].y / 640f;
            float nW = boxes[i].z / 640f;
            float nH = boxes[i].w / 640f;

            rt.anchoredPosition = new Vector2(nX * w, (1 - nY - nH) * h);
            rt.sizeDelta = new Vector2(nW * w, nH * h);

            // [新增] 設定文字
            int id = classes[i];
            string name = (id >= 0 && id < classNames.Length) ? classNames[id] : "Unknown";

            // 尋找 Prefab 裡的 TextMeshPro 元件
            TMP_Text textComponent = box.GetComponentInChildren<TMP_Text>();
            if (textComponent != null)
            {
                textComponent.text = name;
            }
        }
    }

    void OnDestroy()
    {
        worker?.Dispose();
        inputTensor?.Dispose();
        boxBuf?.Release(); classBuf?.Release(); scoreBuf?.Release(); countBuf?.Release();
    }
}