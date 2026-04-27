using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using AntiInterference2D.GUI.Models;

namespace AntiInterference2D.GUI.Services;

/// <summary>
/// FastAPI 后端 HTTP 客户端
/// </summary>
public class ApiClient
{
    private readonly HttpClient _httpClient;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public string BaseAddress => _httpClient.BaseAddress?.ToString() ?? "null";

    public ApiClient(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    public void SetBaseAddress(string baseUrl)
    {
        _httpClient.BaseAddress = new Uri(baseUrl.TrimEnd('/'));
    }

    // ========== 健康检查 ==========
    public async Task<bool> HealthCheckAsync(CancellationToken ct = default)
    {
        try
        {
            var response = await _httpClient.GetAsync("/api/v1/health", ct);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    // ========== 系统状态 ==========
    public async Task<SystemStatus?> GetStatusAsync(CancellationToken ct = default)
    {
        var response = await _httpClient.GetAsync("/api/v1/status", ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<SystemStatus>(JsonOptions, ct);
    }

    // ========== 模型管理 ==========
    public async Task<List<ModelInfo>?> ListModelsAsync(CancellationToken ct = default)
    {
        var response = await _httpClient.GetAsync("/api/v1/models", ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<List<ModelInfo>>(JsonOptions, ct);
    }

    public async Task<bool> SwitchModelAsync(string modelPath, string modelType = "FLARE", string? modelArch = null, CancellationToken ct = default)
    {
        var dict = new Dictionary<string, string>
        {
            ["model_path"] = modelPath,
            ["model_type"] = modelType,
        };
        if (!string.IsNullOrEmpty(modelArch))
            dict["model_arch"] = modelArch;
        var content = new FormUrlEncodedContent(dict);
        var response = await _httpClient.PostAsync("/api/v1/models/switch", content, ct);
        return response.IsSuccessStatusCode;
    }

    // ========== 配置管理 ==========
    public async Task<InferenceConfig?> GetConfigAsync(CancellationToken ct = default)
    {
        var response = await _httpClient.GetAsync("/api/v1/config", ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<InferenceConfig>(JsonOptions, ct);
    }

    public async Task<bool> UpdateConfigAsync(InferenceConfig config, CancellationToken ct = default)
    {
        var response = await _httpClient.PostAsJsonAsync("/api/v1/config", config, JsonOptions, ct);
        return response.IsSuccessStatusCode;
    }

    // ========== 单张图像推理（文件上传） ==========
    public async Task<InferResult?> InferImageAsync(
        string imagePath,
        bool useHdr = true,
        bool useHighlightRepair = true,
        float segThreshold = 0.5f,
        float edgeThreshold = 0.3f,
        bool returnVis = true,
        bool returnCoords = true,
        CancellationToken ct = default)
    {
        await using var fs = File.OpenRead(imagePath);
        var content = new MultipartFormDataContent();
        var streamContent = new StreamContent(fs);
        streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/png");
        content.Add(streamContent, "file", Path.GetFileName(imagePath));
        content.Add(new StringContent(useHdr.ToString().ToLower()), "use_hdr");
        content.Add(new StringContent(useHighlightRepair.ToString().ToLower()), "use_highlight_repair");
        content.Add(new StringContent(segThreshold.ToString()), "seg_threshold");
        content.Add(new StringContent(edgeThreshold.ToString()), "edge_threshold");
        content.Add(new StringContent(returnVis.ToString().ToLower()), "return_visualization");
        content.Add(new StringContent(returnCoords.ToString().ToLower()), "return_coordinates");

        var response = await _httpClient.PostAsync("/api/v1/infer/image", content, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<InferResult>(JsonOptions, ct);
    }

    // ========== Base64 推理（内存图像） ==========
    public async Task<InferResult?> InferBase64Async(
        string base64Image,
        bool useHdr = true,
        float segThreshold = 0.5f,
        float edgeThreshold = 0.3f,
        bool returnVis = true,
        CancellationToken ct = default)
    {
        var content = new FormUrlEncodedContent(new Dictionary<string, string>
        {
            ["image_b64"] = base64Image,
            ["use_hdr"] = useHdr.ToString().ToLower(),
            ["seg_threshold"] = segThreshold.ToString(),
            ["edge_threshold"] = edgeThreshold.ToString(),
            ["return_visualization"] = returnVis.ToString().ToLower(),
        });

        var response = await _httpClient.PostAsync("/api/v1/infer/base64", content, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<InferResult>(JsonOptions, ct);
    }

    // ========== 批量推理 ==========
    public async Task<BatchInferResult?> InferBatchAsync(
        List<string> imagePaths,
        bool useHdr = true,
        float segThreshold = 0.5f,
        float edgeThreshold = 0.3f,
        CancellationToken ct = default)
    {
        var content = new MultipartFormDataContent();
        foreach (var path in imagePaths)
        {
            await using var fs = File.OpenRead(path);
            var streamContent = new StreamContent(fs);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/png");
            content.Add(streamContent, "files", Path.GetFileName(path));
        }
        content.Add(new StringContent(useHdr.ToString().ToLower()), "use_hdr");
        content.Add(new StringContent(segThreshold.ToString()), "seg_threshold");
        content.Add(new StringContent(edgeThreshold.ToString()), "edge_threshold");

        var response = await _httpClient.PostAsync("/api/v1/infer/batch", content, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<BatchInferResult>(JsonOptions, ct);
    }
}
