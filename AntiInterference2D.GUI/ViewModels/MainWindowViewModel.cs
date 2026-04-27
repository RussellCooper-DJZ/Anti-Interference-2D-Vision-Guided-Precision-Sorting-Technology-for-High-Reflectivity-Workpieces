using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using AntiInterference2D.GUI.Models;
using AntiInterference2D.GUI.Services;

namespace AntiInterference2D.GUI.ViewModels;

public partial class MainWindowViewModel : ObservableObject
{
    private readonly ApiClient _api;
    private CancellationTokenSource? _cts;
    private CancellationTokenSource? _pollCts;
    private string? _currentImagePath;
    private bool _continuousRunning;

    // ===== Status =====
    [ObservableProperty] private bool _isConnected;
    [ObservableProperty] private string _backendStatus = "Offline";
    [ObservableProperty] private bool _isRunning;
    [ObservableProperty] private BitmapImage? _displayImage;
    [ObservableProperty] private string _resultStatus = "Ready";
    [ObservableProperty] private string _resultDetails = "No image loaded";
    [ObservableProperty] private string _statusBarText = "";

    // ===== Image Zoom =====
    [ObservableProperty] private double _imageScale = 1.0;
    [ObservableProperty] private string _zoomText = "Zoom: 100%";

    // ===== Stats =====
    [ObservableProperty] private int _okCount;
    [ObservableProperty] private int _ngCount;

    // ===== System Status =====
    [ObservableProperty] private string _systemDevice = "—";
    [ObservableProperty] private string _systemCuda = "—";
    [ObservableProperty] private string _systemMemory = "—";
    [ObservableProperty] private string _systemModel = "—";

    // ===== Log =====
    [ObservableProperty] private string _logText = "";
    [ObservableProperty] private int _selectedBottomTabIndex;

    // ===== Parameters =====
    [ObservableProperty] private float _segThreshold = 0.5f;
    [ObservableProperty] private float _edgeThreshold = 0.3f;
    [ObservableProperty] private bool _useHdr = true;
    [ObservableProperty] private bool _useHighlightRepair = true;
    [ObservableProperty] private bool _returnVis = true;
    [ObservableProperty] private string _selectedBackend = "pytorch";
    [ObservableProperty] private string _selectedDevice = "auto";

    // ===== Data =====
    [ObservableProperty] private ObservableCollection<ResultRow> _resultRows = new();
    [ObservableProperty] private ObservableCollection<ModelInfo> _models = new();
    [ObservableProperty] private ModelInfo? _selectedModelInfo;

    // ===== Options =====
    [ObservableProperty] private ObservableCollection<string> _backendOptions = new() { "pytorch", "onnx", "tensorrt" };

    // ===== Toolbox =====
    [ObservableProperty] private ObservableCollection<ToolItem> _toolItems = new();
    [ObservableProperty] private ToolItem? _selectedTool;

    public MainWindowViewModel(ApiClient api)
    {
        _api = api;

        ToolItems.Add(new ToolItem { Name = "Image", Icon = "📷", Description = "Single image inspection" });
        ToolItems.Add(new ToolItem { Name = "Batch", Icon = "📁", Description = "Batch processing folder" });
        ToolItems.Add(new ToolItem { Name = "Models", Icon = "🧠", Description = "Model management" });
        ToolItems.Add(new ToolItem { Name = "Robot", Icon = "🤖", Description = "Robot communication" });
        ToolItems.Add(new ToolItem { Name = "Config", Icon = "⚙️", Description = "System configuration" });
        SelectedTool = ToolItems[0];

        _ = Task.Run(async () =>
        {
            try
            {
                var ok = await _api.HealthCheckAsync();
                if (ok)
                {
                    IsConnected = true;
                    BackendStatus = "Online";
                    await RefreshModelsAsync();
                    await RefreshSystemStatusAsync();
                }
            }
            catch (Exception ex)
            {
                Log($"Initial health check error: {ex.Message}");
            }
            await StartPollingAsync();
        });
    }

    private void Log(string msg)
    {
        var line = $"[{DateTime.Now:HH:mm:ss}] {msg}{Environment.NewLine}";
        LogText += line;
    }

    // ===== Polling =====
    private async Task StartPollingAsync()
    {
        _pollCts = new CancellationTokenSource();
        while (!_pollCts.Token.IsCancellationRequested)
        {
            try
            {
                var ok = await _api.HealthCheckAsync(_pollCts.Token);
                if (ok && !IsConnected)
                {
                    IsConnected = true;
                    BackendStatus = "Online";
                    Log("Backend connected.");
                    await RefreshModelsAsync();
                }
                else if (!ok && IsConnected)
                {
                    IsConnected = false;
                    BackendStatus = "Offline";
                    Log("Backend disconnected.");
                }

                if (IsConnected)
                    await RefreshSystemStatusAsync();
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex)
            {
                if (!IsConnected)
                    Log($"Health check error: {ex.Message}");
            }
            try { await Task.Delay(3000, _pollCts.Token); } catch { break; }
        }
    }

    private async Task RefreshModelsAsync()
    {
        try
        {
            var list = await _api.ListModelsAsync();
            if (list != null)
            {
                var prevSelected = SelectedModelInfo?.Path;
                Models.Clear();
                foreach (var m in list)
                    Models.Add(m);
                if (prevSelected != null)
                    SelectedModelInfo = Models.FirstOrDefault(m => m.Path == prevSelected);
                if (SelectedModelInfo == null && Models.Count > 0)
                    SelectedModelInfo = Models[0];
            }
        }
        catch (Exception ex)
        {
            Log($"Failed to refresh models: {ex.Message}");
        }
    }

    private async Task RefreshSystemStatusAsync()
    {
        try
        {
            var status = await _api.GetStatusAsync();
            if (status != null)
            {
                SystemDevice = status.Device;
                SystemCuda = status.CudaAvailable ? (status.CudaDeviceName ?? "CUDA") : "CPU";
                SystemMemory = $"{status.MemoryUsedMb:F0}/{status.MemoryTotalMb:F0} MB";
                SystemModel = status.CurrentModel != null ? Path.GetFileName(status.CurrentModel) : "None";
            }
        }
        catch { /* ignore */ }
    }

    // ===== Commands =====

    [RelayCommand]
    private async Task OpenImage()
    {
        var dlg = new OpenFileDialog
        {
            Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp|All|*.*",
            Title = "Select Image"
        };
        if (dlg.ShowDialog() != true) return;

        _currentImagePath = dlg.FileName;
        DisplayImage = LoadBitmap(_currentImagePath);
        ResultStatus = "Image Loaded";
        ResultDetails = Path.GetFileName(_currentImagePath);
        Log($"Opened: {_currentImagePath}");

        await RunOnce();
    }

    [RelayCommand]
    private async Task OpenFolder()
    {
        var dlg = new OpenFolderDialog { Title = "Select Image Folder" };
        if (dlg.ShowDialog() != true) return;

        var folder = dlg.FolderName;
        var files = Directory.EnumerateFiles(folder, "*.*")
            .Where(f => ".png.jpg.jpeg.bmp".Contains(Path.GetExtension(f).ToLower()))
            .ToList();
        if (files.Count == 0)
        {
            Log("No images found in folder.");
            return;
        }

        Log($"Batch: {files.Count} images found.");
        IsRunning = true;
        ResultRows.Clear();
        int idx = 1;
        foreach (var path in files)
        {
            if (_cts?.IsCancellationRequested == true) break;
            _currentImagePath = path;
            DisplayImage = LoadBitmap(path);
            ResultStatus = $"Batch {idx}/{files.Count}";
            await RunOnceInternal();
            idx++;
        }
        IsRunning = false;
        ResultStatus = "Batch Done";
        Log("Batch processing complete.");
    }

    [RelayCommand]
    private async Task RunOnce()
    {
        if (string.IsNullOrEmpty(_currentImagePath))
        {
            Log("No image loaded. Use Open to select an image.");
            return;
        }
        if (!IsConnected)
        {
            Log("Backend not connected. Start the backend first.");
            return;
        }
        await RunOnceInternal();
    }

    private async Task RunOnceInternal()
    {
        IsRunning = true;
        ResultStatus = "Running...";
        Log("Inference started...");

        try
        {
            var result = await _api.InferImageAsync(
                _currentImagePath!,
                useHdr: UseHdr,
                useHighlightRepair: UseHighlightRepair,
                segThreshold: SegThreshold,
                edgeThreshold: EdgeThreshold,
                returnVis: ReturnVis,
                returnCoords: true);

            if (result?.Success == true)
            {
                ResultStatus = "OK";
                ResultDetails = $"Latency: {result.LatencyMs:F1} ms";
                StatusBarText = $"Latency: {result.LatencyMs:F1} ms | Model: {SystemModel}";
                Log($"Inference OK. Latency={result.LatencyMs:F1}ms");
                OkCount++;

                if (!string.IsNullOrEmpty(result.VisImageB64))
                {
                    DisplayImage = Base64ToBitmap(result.VisImageB64);
                }

                ResultRows.Clear();
                if (result.Coordinates != null)
                {
                    int idx = 1;
                    foreach (var c in result.Coordinates)
                    {
                        ResultRows.Add(new ResultRow
                        {
                            ItemName = $"Defect {idx}",
                            X = $"{c.Cx:F1}",
                            Y = $"{c.Cy:F1}",
                            Score = $"{c.Area:F2}",
                            Status = "NG"
                        });
                        idx++;
                    }
                }
                if (result.Metrics != null)
                {
                    foreach (var kv in result.Metrics)
                    {
                        ResultRows.Add(new ResultRow
                        {
                            ItemName = kv.Key,
                            Score = $"{kv.Value:F4}",
                            Status = "Metric"
                        });
                    }
                }
                SelectedBottomTabIndex = 0; // Auto switch to Results
            }
            else
            {
                ResultStatus = "Error";
                ResultDetails = result?.Message ?? "Unknown error";
                Log($"Inference failed: {ResultDetails}");
                NgCount++;
            }
        }
        catch (Exception ex)
        {
            ResultStatus = "Exception";
            ResultDetails = ex.Message;
            Log($"Inference exception: {ex.Message}");
            NgCount++;
        }
        finally
        {
            IsRunning = false;
        }
    }

    [RelayCommand]
    private async Task RunContinuous()
    {
        if (_continuousRunning) return;
        _continuousRunning = true;
        IsRunning = true;
        _cts = new CancellationTokenSource();
        Log("Continuous mode started.");

        while (!_cts.Token.IsCancellationRequested)
        {
            if (!string.IsNullOrEmpty(_currentImagePath))
                await RunOnceInternal();
            try { await Task.Delay(500, _cts.Token); } catch { break; }
        }

        _continuousRunning = false;
        IsRunning = false;
        Log("Continuous mode stopped.");
    }

    [RelayCommand]
    private void Stop()
    {
        _cts?.Cancel();
        _continuousRunning = false;
        IsRunning = false;
        Log("Stop requested.");
    }

    [RelayCommand]
    private async Task LoadModel()
    {
        if (SelectedModelInfo == null)
        {
            Log("No model selected. Please select a model from the dropdown.");
            return;
        }
        if (!IsConnected)
        {
            Log("Backend not connected.");
            return;
        }
        try
        {
            Log($"Loading model: {SelectedModelInfo.Name} ({SelectedModelInfo.ModelArch})...");
            var ok = await _api.SwitchModelAsync(
                SelectedModelInfo.Path,
                SelectedModelInfo.ModelArch,
                SelectedModelInfo.ModelArch);
            if (ok)
            {
                Log("Model loaded successfully.");
                await RefreshModelsAsync();
                await RefreshSystemStatusAsync();
            }
            else
            {
                Log("Model load failed (server returned error).");
            }
        }
        catch (Exception ex)
        {
            Log($"Load model error: {ex.Message}");
        }
    }

    [RelayCommand]
    private async Task ApplyConfig()
    {
        if (!IsConnected)
        {
            Log("Backend not connected.");
            return;
        }
        try
        {
            var config = new InferenceConfig
            {
                SegThreshold = SegThreshold,
                EdgeThreshold = EdgeThreshold,
                UseHdr = UseHdr,
                UseHighlightRepair = UseHighlightRepair,
                ReturnVisualization = ReturnVis,
                Backend = SelectedBackend,
                Device = SelectedDevice
            };
            var ok = await _api.UpdateConfigAsync(config);
            Log(ok ? "Config applied." : "Config apply failed.");
        }
        catch (Exception ex)
        {
            Log($"Config error: {ex.Message}");
        }
    }

    [RelayCommand]
    private void ExportResults()
    {
        if (ResultRows.Count == 0)
        {
            Log("No results to export.");
            return;
        }
        var dlg = new SaveFileDialog
        {
            Filter = "CSV|*.csv|All|*.*",
            FileName = $"results_{DateTime.Now:yyyyMMdd_HHmmss}.csv"
        };
        if (dlg.ShowDialog() != true) return;

        try
        {
            var lines = new System.Collections.Generic.List<string> { "Item,X,Y,Score,Status" };
            foreach (var r in ResultRows)
                lines.Add($"{r.ItemName},{r.X},{r.Y},{r.Score},{r.Status}");
            File.WriteAllLines(dlg.FileName, lines);
            Log($"Results exported to: {dlg.FileName}");
        }
        catch (Exception ex)
        {
            Log($"Export failed: {ex.Message}");
        }
    }

    [RelayCommand]
    private void SaveConfig()
    {
        var dlg = new SaveFileDialog
        {
            Filter = "JSON|*.json|All|*.*",
            FileName = "config.json"
        };
        if (dlg.ShowDialog() != true) return;
        try
        {
            var config = new InferenceConfig
            {
                SegThreshold = SegThreshold,
                EdgeThreshold = EdgeThreshold,
                UseHdr = UseHdr,
                UseHighlightRepair = UseHighlightRepair,
                ReturnVisualization = ReturnVis,
                Backend = SelectedBackend,
                Device = SelectedDevice
            };
            var json = System.Text.Json.JsonSerializer.Serialize(config, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(dlg.FileName, json);
            Log($"Config saved to: {dlg.FileName}");
        }
        catch (Exception ex)
        {
            Log($"Save config failed: {ex.Message}");
        }
    }

    [RelayCommand]
    private void LoadConfig()
    {
        var dlg = new OpenFileDialog
        {
            Filter = "JSON|*.json|All|*.*",
            Title = "Load Config"
        };
        if (dlg.ShowDialog() != true) return;
        try
        {
            var json = File.ReadAllText(dlg.FileName);
            var config = System.Text.Json.JsonSerializer.Deserialize<InferenceConfig>(json);
            if (config != null)
            {
                SegThreshold = config.SegThreshold;
                EdgeThreshold = config.EdgeThreshold;
                UseHdr = config.UseHdr;
                UseHighlightRepair = config.UseHighlightRepair;
                ReturnVis = config.ReturnVisualization;
                SelectedBackend = config.Backend;
                SelectedDevice = config.Device;
                Log("Config loaded.");
            }
        }
        catch (Exception ex)
        {
            Log($"Load config failed: {ex.Message}");
        }
    }

    [RelayCommand]
    private void ZoomIn()
    {
        ImageScale = Math.Min(ImageScale * 1.2, 10.0);
        ZoomText = $"Zoom: {ImageScale * 100:F0}%";
    }

    [RelayCommand]
    private void ZoomOut()
    {
        ImageScale = Math.Max(ImageScale / 1.2, 0.1);
        ZoomText = $"Zoom: {ImageScale * 100:F0}%";
    }

    [RelayCommand]
    private void ResetZoom()
    {
        ImageScale = 1.0;
        ZoomText = "Zoom: 100%";
    }

    public void OnMouseWheel(double delta)
    {
        if (delta > 0)
            ZoomIn();
        else
            ZoomOut();
    }

    [RelayCommand] private void ShowImage()      => SelectedTool = ToolItems.FirstOrDefault(t => t.Name == "Image");
    [RelayCommand] private void ShowBatch()      => SelectedTool = ToolItems.FirstOrDefault(t => t.Name == "Batch");
    [RelayCommand] private void ShowModels()     => SelectedTool = ToolItems.FirstOrDefault(t => t.Name == "Models");
    [RelayCommand] private void ShowRobot()      => SelectedTool = ToolItems.FirstOrDefault(t => t.Name == "Robot");
    [RelayCommand] private void ShowSettings()   => SelectedTool = ToolItems.FirstOrDefault(t => t.Name == "Config");
    [RelayCommand] private void ShowLogs()       => SelectedBottomTabIndex = 1;
    [RelayCommand] private void ExitApplication() => Application.Current.Shutdown();

    // ===== Helpers =====
    private static BitmapImage LoadBitmap(string path)
    {
        var bmp = new BitmapImage();
        bmp.BeginInit();
        bmp.UriSource = new Uri(path, UriKind.Absolute);
        bmp.CacheOption = BitmapCacheOption.OnLoad;
        bmp.EndInit();
        bmp.Freeze();
        return bmp;
    }

    private static BitmapImage Base64ToBitmap(string b64)
    {
        var bytes = Convert.FromBase64String(b64);
        var bmp = new BitmapImage();
        bmp.BeginInit();
        bmp.StreamSource = new MemoryStream(bytes);
        bmp.CacheOption = BitmapCacheOption.OnLoad;
        bmp.EndInit();
        bmp.Freeze();
        return bmp;
    }
}
