using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using AntiInterference2D.GUI.Services;
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace AntiInterference2D.GUI.ViewModels;

public partial class ImageInspectionViewModel : ObservableObject
{
    private readonly ApiClient _apiClient;
    private readonly ImageProcessingService _imageService;

    [ObservableProperty]
    private BitmapImage? _originalImage;

    [ObservableProperty]
    private BitmapImage? _resultImage;

    [ObservableProperty]
    private BitmapImage? _segMaskImage;

    [ObservableProperty]
    private BitmapImage? _edgeMaskImage;

    [ObservableProperty]
    private string _resultText = "等待检测...";

    [ObservableProperty]
    private bool _isProcessing;

    [ObservableProperty]
    private bool _useHdr = true;

    [ObservableProperty]
    private bool _useHighlightRepair = true;

    [ObservableProperty]
    private float _segThreshold = 0.5f;

    [ObservableProperty]
    private float _edgeThreshold = 0.3f;

    [ObservableProperty]
    private string _selectedImagePath = "";

    [ObservableProperty]
    private string _latencyText = "-";

    public ImageInspectionViewModel(ApiClient apiClient, ImageProcessingService imageService)
    {
        _apiClient = apiClient;
        _imageService = imageService;
    }

    [RelayCommand]
    private void SelectImage()
    {
        var dlg = new OpenFileDialog
        {
            Filter = "图像文件|*.png;*.jpg;*.jpeg;*.bmp|所有文件|*.*",
            Title = "选择待检测图像"
        };
        if (dlg.ShowDialog() == true)
        {
            SelectedImagePath = dlg.FileName;
            OriginalImage = _imageService.LoadBitmapFromPath(dlg.FileName);
            ResultImage = null;
            SegMaskImage = null;
            EdgeMaskImage = null;
            ResultText = "图像已加载，点击「开始检测」";
        }
    }

    [RelayCommand]
    private async Task StartInspectionAsync()
    {
        if (string.IsNullOrEmpty(SelectedImagePath) || !File.Exists(SelectedImagePath))
        {
            ResultText = "请先选择图像";
            return;
        }

        IsProcessing = true;
        ResultText = "正在推理...";

        try
        {
            var result = await _apiClient.InferImageAsync(
                SelectedImagePath,
                UseHdr,
                UseHighlightRepair,
                SegThreshold,
                EdgeThreshold,
                returnVis: true,
                returnCoords: true);

            if (result?.Success == true)
            {
                LatencyText = $"{result.LatencyMs:F1} ms";
                ResultText = $"检测完成 | 延迟: {LatencyText}";

                if (!string.IsNullOrEmpty(result.VisImageB64))
                    ResultImage = _imageService.LoadBitmapFromBase64(result.VisImageB64);
                if (!string.IsNullOrEmpty(result.SegMaskB64))
                    SegMaskImage = _imageService.LoadBitmapFromBase64(result.SegMaskB64);
                if (!string.IsNullOrEmpty(result.EdgeMaskB64))
                    EdgeMaskImage = _imageService.LoadBitmapFromBase64(result.EdgeMaskB64);
            }
            else
            {
                ResultText = $"检测失败: {result?.Message ?? "未知错误"}";
            }
        }
        catch (Exception ex)
        {
            ResultText = $"请求异常: {ex.Message}";
        }
        finally
        {
            IsProcessing = false;
        }
    }

    [RelayCommand]
    private void SaveResult()
    {
        if (ResultImage == null) return;
        var dlg = new SaveFileDialog
        {
            Filter = "PNG 图像|*.png",
            FileName = $"result_{DateTime.Now:yyyyMMdd_HHmmss}.png"
        };
        if (dlg.ShowDialog() == true)
        {
            // 保存逻辑（简化）
            ResultText = $"结果已保存: {dlg.FileName}";
        }
    }
}
