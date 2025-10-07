# Font Standardization - Horizontal Layout

## Issues Fixed

### 1. Negative Font Sizes (QFont::setPointSize Error)
**Problem**: Four buttons had `<pointsize>-1</pointsize>` causing Qt warning
**Location**: Curation tab - moveAllRight, moveSingleRight, moveSingleLeft, moveAllLeft buttons

**Fix**: Changed all from -1 to 14pt to match their stylesheet font-size

### 2. Mixed Fonts in Groupboxes
**Problem**: Widgets within groupboxes had inconsistent fonts despite appearing the same in Qt Designer
**Cause**: Some widgets inherited bold fonts from parent, others had explicit font settings

**Solution**: Standardized ALL fonts within the three main groupboxes to match "Analyze:" label

## Standard Font

All widgets within groupboxes now use:
```xml
<property name="font">
 <font>
  <pointsize>9</pointsize>
  <bold>false</bold>
 </font>
</property>
```

## Groupboxes Affected

### 1. Channel Selection (groupBox_2)
- AnalyzeLabel
- StimChanSelectLabel

### 2. Filtering & Preprocessing (groupBox)
- LowPass_checkBox
- HighPass_checkBox
- FilterOrderLabel
- InvertSignal_checkBox
- SpectralAnalysisButton (link style button)

### 3. Peak Detection (PeakDetection)
- ThresholdLabel
- PeakProminenceLabel
- PeakDistanceLabel

## Widgets NOT Changed

- **Groupbox Titles**: Remain 10pt bold (e.g., "Channel Selection", "Filtering & Preprocessing", "Peak Detection")
- **Button Move Icons**: Remain 14pt bold (>>, >, <, <<)
- **Widgets Outside Groupboxes**: Not affected by this standardization

## Changes Summary

- **Fixed**: 4 negative font sizes (-1 → 14pt)
- **Standardized**: 10 fonts within groupboxes (all → 9pt non-bold)
- **Total Changes**: 14 font modifications

## Result

✅ No more font size warnings
✅ Consistent appearance within all groupboxes
✅ Clean 9pt non-bold font matching "Analyze:" label
✅ Professional, unified look

## Script Used

`standardize_groupbox_fonts.py` - Automated font standardization tool
- Locates groupbox boundaries
- Preserves groupbox title fonts (10pt bold)
- Standardizes all child widget fonts to 9pt non-bold
- Safe and repeatable
