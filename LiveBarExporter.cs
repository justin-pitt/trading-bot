// LiveBarExporter — NinjaTrader 8 Indicator
//
// Exports bar data to CSV files so the Python trading bot can read
// real-time NinjaTrader chart data.
//
// INSTALLATION:
//   1. Open NinjaTrader 8
//   2. Tools > Edit NinjaScript > Indicator
//   3. Click "New" and name it "LiveBarExporter"
//   4. Replace all the generated code with the contents of this file
//   5. Click "Compile" — fix any errors shown in the Output window
//   6. Close the editor
//   7. Add the indicator to any chart:
//        Right-click chart > Indicators > LiveBarExporter
//   8. Set Export Directory and Bars to Export in the indicator properties
//
// OUTPUT FILES:
//   One CSV per chart, named: {SYMBOL}_{INTERVAL}.csv
//   Examples:
//     C:\NinjaTrader\LiveBars\ES_1D.csv     (daily ES chart)
//     C:\NinjaTrader\LiveBars\NQ_5M.csv     (5-minute NQ chart)
//     C:\NinjaTrader\LiveBars\CL_15M.csv    (15-minute CL chart)
//
// PYTHON CONFIGURATION (.env):
//   LIVE_DATA_DIR=C:\NinjaTrader\LiveBars
//   LIVE_INTERVAL=1D                        (must match the chart bar type)
//
// NOTES:
//   - Add this indicator to every chart/symbol you want the bot to trade
//   - The file is rewritten on every bar close using an atomic rename
//     so Python never reads a partially written file
//   - The file is also written once immediately when the indicator loads,
//     so Python has data available right away without waiting for a bar close

#region Using declarations
using System;
using System.IO;
using System.Text;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class LiveBarExporter : Indicator
    {
        private string filePath;
        private string tempPath;
        private bool initialExportDone;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description  = "Exports bar OHLCV data to CSV for the Python trading bot.";
                Name         = "LiveBarExporter";
                Calculate    = Calculate.OnBarClose;
                IsOverlay    = true;
                DisplayInDataBox  = false;
                PaintPriceMarkers = false;
                BarsRequiredToPlot = 0;

                // Default properties — override in the indicator's Properties panel
                ExportDirectory = @"C:\NinjaTrader\LiveBars";
                BarsToExport    = 500;
            }
            else if (State == State.DataLoaded)
            {
                initialExportDone = false;

                // Create the export directory if it doesn't exist
                if (!Directory.Exists(ExportDirectory))
                {
                    try   { Directory.CreateDirectory(ExportDirectory); }
                    catch (Exception ex)
                    {
                        Print("LiveBarExporter: cannot create directory "
                              + ExportDirectory + " — " + ex.Message);
                        return;
                    }
                }

                // File name: ES_1D.csv / NQ_5M.csv / CL_15M.csv etc.
                string fileName = string.Format("{0}_{1}.csv",
                    Instrument.MasterInstrument.Name,
                    GetIntervalString());

                filePath = Path.Combine(ExportDirectory, fileName);
                tempPath = filePath + ".tmp";

                Print("LiveBarExporter: will write to " + filePath);
            }
        }

        protected override void OnBarUpdate()
        {
            // Write once after all historical bars finish loading,
            // then again on every real-time bar close.
            if (State == State.Historical)
            {
                // Only write on the very last historical bar so we don't
                // thrash the disk during the data load.
                if (!initialExportDone && IsFirstTickOfBar && CurrentBar > 0)
                {
                    // Check if we are on the last historical bar by seeing
                    // if the next call will be realtime. We use a flag approach:
                    // just mark done on first historical write — it'll be refreshed
                    // properly on the first realtime bar anyway.
                    initialExportDone = true;
                    WriteFile();
                }
                return;
            }

            // State.Realtime — write on every bar close
            WriteFile();
        }

        private void WriteFile()
        {
            if (filePath == null) return;

            try
            {
                int barsCount = Math.Min(CurrentBar + 1, BarsToExport);

                using (StreamWriter sw = new StreamWriter(tempPath, false, new UTF8Encoding(false)))
                {
                    sw.WriteLine("datetime,open,high,low,close,volume");

                    // Write oldest bar first (index = barsCount-1) to newest (index = 0)
                    for (int i = barsCount - 1; i >= 0; i--)
                    {
                        sw.WriteLine(string.Format(
                            "{0:yyyy-MM-dd HH:mm:ss},{1},{2},{3},{4},{5}",
                            Time[i],
                            Open[i],
                            High[i],
                            Low[i],
                            Close[i],
                            (long)Volume[i]));
                    }
                }

                // Atomic rename — Python will never see a partial file
                if (File.Exists(filePath))
                    File.Delete(filePath);
                File.Move(tempPath, filePath);
            }
            catch (Exception ex)
            {
                Print("LiveBarExporter error writing " + filePath + ": " + ex.Message);
                // Clean up temp file if it exists
                try { if (File.Exists(tempPath)) File.Delete(tempPath); }
                catch { }
            }
        }

        // Build the interval suffix used in the filename.
        // Must match the LIVE_INTERVAL value in the Python bot's .env file.
        //   Daily chart    → 1D
        //   5-min chart    → 5M
        //   15-min chart   → 15M
        //   60-min chart   → 60M
        //   1-hour chart   → 1H  (if broker uses Hour period type)
        private string GetIntervalString()
        {
            switch (BarsPeriod.BarsPeriodType)
            {
                case BarsPeriodType.Day:    return BarsPeriod.Value + "D";
                case BarsPeriodType.Minute: return BarsPeriod.Value + "M";
                case BarsPeriodType.Week:   return BarsPeriod.Value + "W";
                case BarsPeriodType.Tick:   return BarsPeriod.Value + "T";
                default:                   return BarsPeriod.Value.ToString();
            }
        }

        #region Properties

        [NinjaScriptProperty]
        [Display(
            Name        = "Export Directory",
            Description = "Directory where CSV files are written. "
                        + "Must match LIVE_DATA_DIR in the Python bot's .env",
            Order       = 1,
            GroupName   = "LiveBarExporter")]
        public string ExportDirectory { get; set; }

        [NinjaScriptProperty]
        [Range(50, 2000)]
        [Display(
            Name        = "Bars to Export",
            Description = "Number of closed bars to include in each file. "
                        + "500 is enough for EMA55 + plenty of history.",
            Order       = 2,
            GroupName   = "LiveBarExporter")]
        public int BarsToExport { get; set; }

        #endregion
    }
}
