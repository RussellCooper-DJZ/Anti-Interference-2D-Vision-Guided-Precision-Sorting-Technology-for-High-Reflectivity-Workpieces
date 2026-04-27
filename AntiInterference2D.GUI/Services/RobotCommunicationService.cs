using System;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace AntiInterference2D.GUI.Services;

/// <summary>
/// ABB 机器人通信服务（TCP 客户端）
/// 协议与 Python 端 AbbRobotStudioSim 兼容（JSON over TCP）
/// </summary>
public class RobotCommunicationService
{
    private TcpClient? _client;
    private NetworkStream? _stream;
    private readonly object _lock = new();

    public bool IsConnected => _client?.Connected ?? false;
    public string? LastError { get; private set; }

    public async Task<bool> ConnectAsync(string host, int port, CancellationToken ct = default)
    {
        try
        {
            Disconnect();
            _client = new TcpClient();
            await _client.ConnectAsync(host, port, ct);
            _stream = _client.GetStream();
            LastError = null;
            return true;
        }
        catch (Exception ex)
        {
            LastError = ex.Message;
            return false;
        }
    }

    public void Disconnect()
    {
        lock (_lock)
        {
            _stream?.Close();
            _client?.Close();
            _stream = null;
            _client = null;
        }
    }

    public async Task<bool> SendTargetAsync(double x, double y, double z, double rx, double ry, double rz, CancellationToken ct = default)
    {
        if (_stream == null) return false;
        var payload = new { cmd = "moveL", target = new[] { x, y, z, rx, ry, rz } };
        var json = JsonSerializer.Serialize(payload) + "\n";
        var bytes = Encoding.UTF8.GetBytes(json);
        try
        {
            await _stream.WriteAsync(bytes, ct);
            return true;
        }
        catch (Exception ex)
        {
            LastError = ex.Message;
            return false;
        }
    }

    public async Task<string?> GetStatusAsync(CancellationToken ct = default)
    {
        if (_stream == null) return null;
        var payload = new { cmd = "status" };
        var json = JsonSerializer.Serialize(payload) + "\n";
        var bytes = Encoding.UTF8.GetBytes(json);
        try
        {
            await _stream.WriteAsync(bytes, ct);
            var buffer = new byte[4096];
            int read = await _stream.ReadAsync(buffer, ct);
            return Encoding.UTF8.GetString(buffer, 0, read);
        }
        catch (Exception ex)
        {
            LastError = ex.Message;
            return null;
        }
    }
}
