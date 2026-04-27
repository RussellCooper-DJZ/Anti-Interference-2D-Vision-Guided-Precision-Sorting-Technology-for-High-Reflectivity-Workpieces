using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace AntiInterference2D.GUI.Models;

public class ToolItem
{
    public string Name { get; set; } = "";
    public string Icon { get; set; } = "◆";
    public string Description { get; set; } = "";
}

public class ResultRow
{
    public string ItemName { get; set; } = "";
    public string X { get; set; } = "";
    public string Y { get; set; } = "";
    public string Score { get; set; } = "";
    public string Status { get; set; } = "";
}

// ========== 推理配置 ==========
public class InferenceConfig
{
    [JsonPropertyName("img_size")]
    public int ImgSize { get; set; } = 512;

    [JsonPropertyName("use_hdr")]
    public bool UseHdr { get; set; } = true;

    [JsonPropertyName("use_highlight_repair")]
    public bool UseHighlightRepair { get; set; } = true;

    [JsonPropertyName("seg_threshold")]
    public float SegThreshold { get; set; } = 0.5f;

    [JsonPropertyName("edge_threshold")]
    public float EdgeThreshold { get; set; } = 0.3f;

    [JsonPropertyName("device")]
    public string Device { get; set; } = "auto";

    [JsonPropertyName("backend")]
    public string Backend { get; set; } = "pytorch";

    [JsonPropertyName("return_visualization")]
    public bool ReturnVisualization { get; set; } = true;

    [JsonPropertyName("return_coordinates")]
    public bool ReturnCoordinates { get; set; } = true;
}

// ========== 模型信息 ==========
public class ModelInfo
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("path")]
    public string Path { get; set; } = "";

    [JsonPropertyName("type")]
    public string Type { get; set; } = "";

    [JsonPropertyName("model_arch")]
    public string ModelArch { get; set; } = "FLARE";

    [JsonPropertyName("size_mb")]
    public float SizeMb { get; set; }

    [JsonPropertyName("loaded")]
    public bool Loaded { get; set; }
}

// ========== 推理结果 ==========
public class InferResult
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }

    [JsonPropertyName("message")]
    public string Message { get; set; } = "";

    [JsonPropertyName("latency_ms")]
    public float LatencyMs { get; set; }

    [JsonPropertyName("seg_mask_b64")]
    public string? SegMaskB64 { get; set; }

    [JsonPropertyName("edge_mask_b64")]
    public string? EdgeMaskB64 { get; set; }

    [JsonPropertyName("highlight_mask_b64")]
    public string? HighlightMaskB64 { get; set; }

    [JsonPropertyName("vis_image_b64")]
    public string? VisImageB64 { get; set; }

    [JsonPropertyName("coordinates")]
    public List<CoordinateInfo>? Coordinates { get; set; }

    [JsonPropertyName("metrics")]
    public Dictionary<string, float>? Metrics { get; set; }
}

public class CoordinateInfo
{
    [JsonPropertyName("cx")]
    public float Cx { get; set; }

    [JsonPropertyName("cy")]
    public float Cy { get; set; }

    [JsonPropertyName("area")]
    public float Area { get; set; }

    [JsonPropertyName("orientation")]
    public float Orientation { get; set; }

    [JsonPropertyName("x_mm")]
    public float Xmm { get; set; }

    [JsonPropertyName("y_mm")]
    public float Ymm { get; set; }

    [JsonPropertyName("z_mm")]
    public float Zmm { get; set; }
}

// ========== 批量推理结果 ==========
public class BatchInferResult
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }

    [JsonPropertyName("results")]
    public List<BatchItemResult>? Results { get; set; }
}

public class BatchItemResult
{
    [JsonPropertyName("filename")]
    public string Filename { get; set; } = "";

    [JsonPropertyName("success")]
    public bool Success { get; set; }

    [JsonPropertyName("error")]
    public string? Error { get; set; }

    [JsonPropertyName("seg_ratio")]
    public float SegRatio { get; set; }

    [JsonPropertyName("edge_ratio")]
    public float EdgeRatio { get; set; }
}

// ========== 系统状态 ==========
public class SystemStatus
{
    [JsonPropertyName("ready")]
    public bool Ready { get; set; }

    [JsonPropertyName("current_model")]
    public string? CurrentModel { get; set; }

    [JsonPropertyName("current_model_type")]
    public string CurrentModelType { get; set; } = "";

    [JsonPropertyName("device")]
    public string Device { get; set; } = "";

    [JsonPropertyName("cuda_available")]
    public bool CudaAvailable { get; set; }

    [JsonPropertyName("cuda_device_name")]
    public string? CudaDeviceName { get; set; }

    [JsonPropertyName("memory_used_mb")]
    public float MemoryUsedMb { get; set; }

    [JsonPropertyName("memory_total_mb")]
    public float MemoryTotalMb { get; set; }
}
