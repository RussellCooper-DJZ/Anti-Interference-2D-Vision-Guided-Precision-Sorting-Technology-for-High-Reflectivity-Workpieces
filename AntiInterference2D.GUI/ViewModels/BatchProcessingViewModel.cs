using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using AntiInterference2D.GUI.Services;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace AntiInterference2D.GUI.ViewModels;

public partial class BatchProcessingViewModel : ObservableObject
{
    private readonly ApiClient _apiClient;

    [ObservableProperty]
    private string _folderPath = "";

    [ObservableProperty]
    private ObservableCollection<BatchItemViewModel> _items = new();

    [ObservableProperty]
    private bool _isProcessing;

    [ObservableProperty]
    private int _processedCount;

    [ObservableProperty]
    private int _totalCount;

    [ObservableProperty]
    private string _progressText = "0 / 0";

    [ObservableProperty]
    private float _segThreshold = 0.5f;

    [ObservableProperty]
    private float _edgeThreshold = 0.3f;

    [ObservableProperty]
    private bool _useHdr = true;

    public BatchProcessingViewModel(ApiClient apiClient)
    {
        _apiClient = apiClient;
    }

    [RelayCommand]
    private void SelectFolder()
    {
        var dlg = new OpenFolderDialog
        {
            Title = "选择包含图像的文件夹"
        };
        if (dlg.ShowDialog() == true)
        {
            FolderPath = dlg.FolderName;
            var files = Directory.EnumerateFiles(FolderPath, "*.*", SearchOption.TopDirectoryOnly)
                .Where(f => f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
                         || f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                         || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)
                         || f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase))
                .ToList();

            Items.Clear();
            foreach (var file in files)
            {
                Items.Add(new BatchItemViewModel { FileName = Path.GetFileName(file), FullPath = file });
            }
            TotalCount = Items.Count;
            ProgressText = $"0 / {TotalCount}";
        }
    }

    [RelayCommand]
    private async Task StartBatchAsync()
    {
        if (Items.Count == 0) return;

        IsProcessing = true;
        ProcessedCount = 0;

        var paths = Items.Select(i => i.FullPath).ToList();
        try
        {
            var result = await _apiClient.InferBatchAsync(paths, UseHdr, SegThreshold, EdgeThreshold);
            if (result?.Success == true && result.Results != null)
            {
                for (int i = 0; i < result.Results.Count && i < Items.Count; i++)
                {
                    var r = result.Results[i];
                    Items[i].Success = r.Success;
                    Items[i].Status = r.Success
                        ? $"Seg={r.SegRatio:F3} Edge={r.EdgeRatio:F3}"
                        : $"失败: {r.Error}";
                    ProcessedCount++;
                    ProgressText = $"{ProcessedCount} / {TotalCount}";
                }
            }
        }
        catch (Exception)
        {
            // 批量错误处理
        }
        finally
        {
            IsProcessing = false;
        }
    }

    [RelayCommand]
    private void ExportResults()
    {
        var dlg = new SaveFileDialog
        {
            Filter = "CSV 文件|*.csv",
            FileName = $"batch_result_{DateTime.Now:yyyyMMdd_HHmmss}.csv"
        };
        if (dlg.ShowDialog() == true)
        {
            var lines = new List<string> { "Filename,Success,Status" };
            lines.AddRange(Items.Select(i => $"{i.FileName},{i.Success},\"{i.Status}\""));
            File.WriteAllLines(dlg.FileName, lines);
        }
    }
}

public partial class BatchItemViewModel : ObservableObject
{
    [ObservableProperty]
    private string _fileName = "";

    [ObservableProperty]
    private string _fullPath = "";

    [ObservableProperty]
    private bool _success;

    [ObservableProperty]
    private string _status = "待处理";
}
