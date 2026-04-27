using CommunityToolkit.Mvvm.ComponentModel;
using AntiInterference2D.GUI.Services;
using System.Threading.Tasks;

namespace AntiInterference2D.GUI.ViewModels;

public partial class DashboardViewModel : ObservableObject
{
    private readonly ApiClient _apiClient;

    [ObservableProperty]
    private string _systemStatus = "未连接";

    [ObservableProperty]
    private string _modelName = "-";

    [ObservableProperty]
    private string _deviceName = "-";

    [ObservableProperty]
    private string _memoryUsage = "-";

    [ObservableProperty]
    private int _totalDetections;

    [ObservableProperty]
    private float _averageLatencyMs;

    [ObservableProperty]
    private bool _isLoading;

    public DashboardViewModel(ApiClient apiClient)
    {
        _apiClient = apiClient;
        _ = LoadStatusAsync();
    }

    public async Task LoadStatusAsync()
    {
        IsLoading = true;
        try
        {
            var status = await _apiClient.GetStatusAsync();
            if (status != null)
            {
                SystemStatus = status.Ready ? "就绪" : "模型未加载";
                ModelName = status.CurrentModel ?? "未加载";
                DeviceName = status.CudaAvailable ? $"CUDA: {status.CudaDeviceName}" : "CPU";
                MemoryUsage = $"{status.MemoryUsedMb:F1} / {status.MemoryTotalMb:F1} MB";
            }
            else
            {
                SystemStatus = "连接失败";
            }
        }
        catch
        {
            SystemStatus = "连接失败";
        }
        finally
        {
            IsLoading = false;
        }
    }
}
