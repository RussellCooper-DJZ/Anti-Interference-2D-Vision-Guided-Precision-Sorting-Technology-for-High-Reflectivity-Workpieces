using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace AntiInterference2D.GUI.Services;

/// <summary>
/// 图像处理辅助服务（本地 WPF 端）
/// </summary>
public class ImageProcessingService
{
    /// <summary>
    /// 将文件路径加载为 BitmapImage
    /// </summary>
    public BitmapImage? LoadBitmapFromPath(string path)
    {
        if (!File.Exists(path)) return null;
        try
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = new Uri(path, UriKind.Absolute);
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.EndInit();
            bitmap.Freeze();
            return bitmap;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// 将 Base64 字符串加载为 BitmapImage
    /// </summary>
    public BitmapImage? LoadBitmapFromBase64(string base64)
    {
        try
        {
            var bytes = Convert.FromBase64String(base64);
            using var ms = new MemoryStream(bytes);
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.StreamSource = ms;
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.EndInit();
            bitmap.Freeze();
            return bitmap;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// 将 BitmapImage 编码为 PNG Base64
    /// </summary>
    public string? EncodeBitmapToBase64(BitmapImage bitmap)
    {
        try
        {
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(bitmap));
            using var ms = new MemoryStream();
            encoder.Save(ms);
            return Convert.ToBase64String(ms.ToArray());
        }
        catch
        {
            return null;
        }
    }
}
