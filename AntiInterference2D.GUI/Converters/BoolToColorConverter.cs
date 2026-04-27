using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace AntiInterference2D.GUI.Converters;

[ValueConversion(typeof(bool), typeof(Brush))]
public class BoolToColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return value is true ? Brushes.LimeGreen : Brushes.OrangeRed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
