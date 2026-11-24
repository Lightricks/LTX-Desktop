# Build Resources

This folder contains resources for building the installer.

## Required Files

Before building the installer, you need to add:

### icon.ico
A Windows icon file (256x256 recommended) for the application.

You can create one from a PNG using online converters like:
- https://convertio.co/png-ico/
- https://icoconvert.com/

Or use ImageMagick:
```powershell
magick convert logo.png -define icon:auto-resize=256,128,64,48,32,16 icon.ico
```

### Optional: installer-sidebar.bmp
A 164x314 BMP image for the NSIS installer sidebar.
