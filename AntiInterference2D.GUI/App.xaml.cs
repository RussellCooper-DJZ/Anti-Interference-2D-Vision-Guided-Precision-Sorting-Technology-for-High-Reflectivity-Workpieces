using Microsoft.Extensions.DependencyInjection;
using System.Windows;
using AntiInterference2D.GUI.Services;
using AntiInterference2D.GUI.ViewModels;

namespace AntiInterference2D.GUI;

public partial class App : Application
{
    public static IServiceProvider ServiceProvider { get; private set; } = null!;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        var services = new ServiceCollection();
        ConfigureServices(services);
        ServiceProvider = services.BuildServiceProvider();
    }

    private static void ConfigureServices(IServiceCollection services)
    {
        // HTTP client
        services.AddHttpClient<ApiClient>(client =>
        {
            client.BaseAddress = new Uri("http://localhost:8000");
            client.Timeout = TimeSpan.FromSeconds(120);
        });

        // Services
        services.AddSingleton<ImageProcessingService>();
        services.AddSingleton<RobotCommunicationService>();

        // ViewModels
        services.AddTransient<MainWindowViewModel>();
        services.AddTransient<ImageInspectionViewModel>();
        services.AddTransient<BatchProcessingViewModel>();
        services.AddTransient<ParameterConfigViewModel>();
        services.AddTransient<ModelManagementViewModel>();
        services.AddTransient<RobotControlViewModel>();
        services.AddTransient<DashboardViewModel>();
    }
}
