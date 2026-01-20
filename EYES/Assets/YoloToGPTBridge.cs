using UnityEngine;
using UnityEngine.UI;
using TMPro; // 引用文字相關庫

public class YoloToGPTBridge : MonoBehaviour
{
    [Header("連接目標")]
    public GPTManager gptManager;   // 拖入 GPTManager
    public RawImage yoloDisplay;    // 拖入顯示相機畫面的 RawImage

    [Header("按鍵設定")]
    // 電腦測試用 (空白鍵)
    public KeyCode debugKey = KeyCode.Space;

    void Update()
    {
        // ---------------------------------------------------------
        // 情況一：電腦鍵盤測試 (按 Space)
        // ---------------------------------------------------------
        if (Input.GetKeyDown(debugKey))
        {
            Debug.Log("【電腦】鍵盤觸發拍照");
            CaptureAndAnalyze();
        }

        // ---------------------------------------------------------
        // 情況二：VR 手把測試 (按右手的 A 鍵)
        // ---------------------------------------------------------
        // OVRInput 是 Meta Quest 專用的輸入指令
        // Button.One 對應右手的 "A" 鍵，或左手的 "X" 鍵
        if (OVRInput.GetDown(OVRInput.Button.One))
        {
            Debug.Log("【VR】手把觸發拍照");
            CaptureAndAnalyze();
        }

        // (選用) 如果你想用食指板機 (Trigger)
        // if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger)) { ... }
    }

    void CaptureAndAnalyze()
    {
        // 1. 安全檢查
        if (yoloDisplay.texture == null)
        {
            Debug.LogError("抓不到相機畫面！請確認 YOLO 已經啟動。");
            return;
        }

        if (gptManager == null)
        {
            Debug.LogError("GPTManager 沒接上！請檢查 Inspector。");
            return;
        }

        // 2. 把當前畫面轉成 Texture2D
        Texture2D snapshot = TextureToTexture2D(yoloDisplay.texture);

        // 3. 傳給 GPTManager
        gptManager.AnalyzeImage(snapshot);
    }

    // 圖片轉換工具
    Texture2D TextureToTexture2D(Texture texture)
    {
        RenderTexture renderTexture = new RenderTexture(texture.width, texture.height, 24);
        RenderTexture currentRT = RenderTexture.active;

        Graphics.Blit(texture, renderTexture);
        RenderTexture.active = renderTexture;

        Texture2D result = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false);
        result.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        result.Apply();

        RenderTexture.active = currentRT;
        Destroy(renderTexture);

        return result;
    }
}