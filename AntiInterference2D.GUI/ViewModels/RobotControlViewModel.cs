using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using AntiInterference2D.GUI.Services;
using System.Threading.Tasks;

namespace AntiInterference2D.GUI.ViewModels;

public partial class RobotControlViewModel : ObservableObject
{
    private readonly RobotCommunicationService _robotService;

    [ObservableProperty]
    private string _robotHost = "127.0.0.1";

    [ObservableProperty]
    private int _robotPort = 10000;

    [ObservableProperty]
    private bool _isConnected;

    [ObservableProperty]
    private string _connectionStatus = "未连接";

    [ObservableProperty]
    private double _targetX;

    [ObservableProperty]
    private double _targetY;

    [ObservableProperty]
    private double _targetZ;

    [ObservableProperty]
    private double _targetRx;

    [ObservableProperty]
    private double _targetRy;

    [ObservableProperty]
    private double _targetRz;

    [ObservableProperty]
    private string _robotResponse = "";

    public RobotControlViewModel(RobotCommunicationService robotService)
    {
        _robotService = robotService;
    }

    [RelayCommand]
    private async Task ConnectAsync()
    {
        var ok = await _robotService.ConnectAsync(RobotHost, RobotPort);
        IsConnected = ok;
        ConnectionStatus = ok ? $"已连接 {RobotHost}:{RobotPort}" : $"连接失败: {_robotService.LastError}";
    }

    [RelayCommand]
    private void Disconnect()
    {
        _robotService.Disconnect();
        IsConnected = false;
        ConnectionStatus = "已断开";
    }

    [RelayCommand]
    private async Task SendTargetAsync()
    {
        if (!IsConnected) return;
        var ok = await _robotService.SendTargetAsync(TargetX, TargetY, TargetZ, TargetRx, TargetRy, TargetRz);
        RobotResponse = ok ? "目标已发送" : $"发送失败: {_robotService.LastError}";
    }

    [RelayCommand]
    private async Task QueryStatusAsync()
    {
        if (!IsConnected) return;
        var resp = await _robotService.GetStatusAsync();
        RobotResponse = resp ?? "无响应";
    }
}
