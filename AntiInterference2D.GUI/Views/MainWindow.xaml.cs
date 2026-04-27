using AntiInterference2D.GUI.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using System.Windows;
using System.Windows.Input;

namespace AntiInterference2D.GUI.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DataContext = App.ServiceProvider.GetRequiredService<MainWindowViewModel>();
    }

    protected override void OnMouseWheel(MouseWheelEventArgs e)
    {
        base.OnMouseWheel(e);
        if (DataContext is MainWindowViewModel vm && vm.DisplayImage != null)
        {
            vm.OnMouseWheel(e.Delta);
            e.Handled = true;
        }
    }
}
