using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using AntiInterference2D.GUI.Services;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;

namespace AntiInterference2D.GUI.ViewModels;

public partial class ModelManagementViewModel : ObservableObject
{
    private readonly ApiClient _apiClient;

    [ObservableProperty]
    private ObservableCollection<Models.ModelInfo> _models = new();

    [ObservableProperty]
    private Models.ModelInfo? _selectedModel;

    [ObservableProperty]
    private string _statusMessage = "";

    [ObservableProperty]
    private bool _isLoading;

    public ModelManagementViewModel(ApiClient apiClient)
    {
        _apiClient = apiClient;
        _ = RefreshModelsAsync();
    }

    [RelayCommand]
    private async Task RefreshModelsAsync()
    {
        IsLoading = true;
        try
        {
            var list = await _apiClient.ListModelsAsync();
            Models.Clear();
            if (list != null)
            {
                foreach (var m in list.OrderByDescending(m => m.Loaded))
                    Models.Add(m);
            }
            StatusMessage = $"发现 {Models.Count} 个模型";
        }
        catch (Exception ex)
        {
            StatusMessage = $"刷新失败: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    private async Task SwitchModelAsync()
    {
        if (SelectedModel == null) return;

        IsLoading = true;
        try
        {
            var ok = await _apiClient.SwitchModelAsync(SelectedModel.Path, SelectedModel.Type);
            StatusMessage = ok ? $"已切换至: {SelectedModel.Name}" : "切换失败";
            if (ok) await RefreshModelsAsync();
        }
        catch (Exception ex)
        {
            StatusMessage = $"切换异常: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }
}
