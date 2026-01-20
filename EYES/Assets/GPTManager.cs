using UnityEngine;
using OpenAI;
using OpenAI.Chat;
using TMPro;
using System.Collections.Generic; // 為了使用 List

public class GPTManager : MonoBehaviour
{
    [Header("設定")]
    public OpenAIConfiguration configuration; // 你的設定檔
    public TextMeshProUGUI resultText;        // UI 文字框

    private OpenAIClient api;

    void Start()
    {
        // 初始化 API
        // 如果 configuration 沒拉，這邊會報錯，記得檢查 Inspector
        if (configuration != null)
        {
            api = new OpenAIClient(configuration);
        }
        else
        {
            Debug.LogError("請在 Inspector 中掛上 OpenAI Configuration 檔案！");
        }
    }

    // 這是給外部呼叫的函式：傳入一張圖片，開始分析
    public async void AnalyzeImage(Texture2D texture)
    {
        if (resultText != null)
        {
            resultText.text = "AI 正在分析畫面...";
        }

        var messages = new List<Message>
        {
            new Message(Role.System, "你是一個智慧眼鏡助手。請用繁體中文，簡短告訴我你看到了什麼物體。"),
            new Message(Role.User, new List<Content>
            {
                "圖片裡有什麼？",
                texture // 套件會自動處理圖片格式
            })
        };

        try
        {
            var chatRequest = new ChatRequest(messages, model: "gpt-4o");
            var response = await api.ChatEndpoint.GetCompletionAsync(chatRequest);

            // --- 修正部分開始 ---
            // 因為新版套件回傳的 Content 可能是 object 格式，所以必須加上 .ToString()
            if (response.FirstChoice.Message.Content != null)
            {
                string reply = response.FirstChoice.Message.Content.ToString();

                if (resultText != null)
                {
                    resultText.text = reply;
                }
                Debug.Log("GPT 回傳: " + reply);
            }
            else
            {
                Debug.LogWarning("GPT 回傳了空訊息");
            }
            // --- 修正部分結束 ---
        }
        catch (System.Exception e)
        {
            if (resultText != null)
            {
                resultText.text = "錯誤: " + e.Message;
            }
            Debug.LogError("OpenAI 錯誤: " + e);
        }
    }
}