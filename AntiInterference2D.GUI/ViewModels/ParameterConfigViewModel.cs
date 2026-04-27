using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using AntiInterference2D.GUI.Services;
using System.Threading.Tasks;

namespace AntiInterference2D.GUI.ViewModels;

public partial class ParameterConfigViewModel : ObservableObject
{
    private readonly ApiClient _apiClient;

    [ObservableProperty]
    private int _imgSize = 512;

    [ObservableProperty]
    private bool _useHdr = true;

    [ObservableProperty]
    private bool _useHighlightRepair = true;

    [ObservableProperty]
    private float _segThreshold = 0.5f;

    [ObservableProperty]
    private float _edgeThreshold = 0.3f;

    [ObservableProperty]
    private string _device = "auto";

    [ObservableProperty]
    private string _backend = "pytorch";

    [ObservableProperty]
    private bool _returnVisualization = true;

    [ObservableProperty]
    private bool _returnCoordinates = true;

    [ObservableProperty]
    private string _statusMessage = "";

    public ParameterConfigViewModel(ApiClient apiClient)
    {
        _apiClient = apiClient;
        _ = LoadConfigAsync();
    }

    public async Task LoadConfigAsync()
    {
        try
        {
            var cfg = await _apiClient.GetConfigAsync();
            if (cfg != null)
            {
                ImgSize = cfg.ImgSize;
                UseHdr = cfg.UseHdr;
                UseHighlightRepair = cfg.UseHighlightRepair;
                SegThreshold = cfg.SegThreshold;
                EdgeThreshold = cfg.EdgeThreshold;
                Device = cfg.Device;
                Backend = cfg.Backend;
                ReturnVisualization = cfg.ReturnVisualization;
                ReturnCoordinates = cfg.ReturnCoordinates;
            }
        }
        catch { /* 忽略加载错误 */ }
    }

    [RelayCommand]
    private async Task SaveConfigAsync()
    {
        var cfg = new Models.InferenceConfig
        {
            ImgSize = ImgSize,
            UseHdr = UseHdr,
            UseHighlightRepair = UseHighlightRepair,
            SegThreshold = SegThreshold,
            EdgeThreshold = EdgeThreshold,
            Device = Device,
            Backend = Backend,
            ReturnVisualization = ReturnVisualization,
            ReturnCoordinates = ReturnCoordinates,
        };

        try
        {
            var ok = await _apiClient.UpdateConfigAsync(cfg);
            StatusMessage = ok ? "配置已保存到后端" : "保存失败";
        }
        catch (System.Exception ex)
        {
            StatusMessage = $"保存异常: {ex.Message}";
        }
    }

    [RelayCommand]
    private void ResetDefaults()
    {
        ImgSize = 512;
        UseHdr = true;
        UseHighlightRepair = true;
        SegThreshold = 0.5f;
        EdgeThreshold = 0.3f;
        Device = "auto";
        Backend = "pytorch";
        ReturnVisualization = true;
        ReturnCoordinates = true;
        StatusMessage = "已重置为默认值";
    }
}
