# Dracula Wallpapers
Wallpapers in the [Dracula](https://draculatheme.com) color scheme, made by recoloring
[linuxdotexe's Nordic Wallpapers](https://github.com/linuxdotexe/nordic-wallpapers) from Nord to Dracula with
[ImageGoNord](https://github.com/Schroedinger-Hat/ImageGoNord-pip) by [Schrödinger Hat](https://github.com/Schroedinger-Hat).

The originals were cherry picked from [reddit](https://reddit.com/r/ImageGoNord), [unsplash](https://unsplash.com/t/wallpapers),
[Christian Chiarulli's discord server](https://discord.gg/Xb9B4Ny), [wallhaven](https://wallhaven.cc),
[ThePrimeagen's wallpaper repository](https://github.com/ThePrimeagen/anime) and
[Linux_Dynamic_Wallpapers](https://github.com/saint-13/Linux_Dynamic_Wallpapers.git).

Previews of the wallpapers can be seen [here](/wallpaper-preview.md), and of the dynamic wallpapers [here](/dynamic-wallpapers-preview.md).

## How they were made
Running ImageGoNord straight on a Nord wallpaper with the 11 Dracula colors flattens it: Nord's four dark shades all land
on one Dracula grey, its three whites on one white, and soft gradients turn into blotches. So `godracula.py` does it in two steps:

1. **A Nord → Dracula color transfer** in the OKLab color space. Each of Nord's 16 colors goes to a chosen Dracula color
   (Polar Night to the Dracula background shades, Snow Storm to Foreground, Frost to Comment, Purple and Cyan,
   Aurora to Red, Orange, Yellow, Green and Pink), keeping Nord's lightness steps, and the colors in between follow smoothly.
2. **ImageGoNord** snaps the result onto the Dracula palette. Photos and soft gradients that were never quantized to Nord
   keep the transfer alone, because quantizing them makes visible bands.

A few wallpapers didn't make the trip: mostly-light images (Dracula is a dark theme), the Formula One photos (real
team liveries), the Nord logo, and a few whose muted Nord colors turned into neon blotches.

### Converting images

```bash
pip install image-go-nord
python godracula.py path/to/image-or-folder ... --out converted/
```

Without `--out` the files are replaced in place. The tests run with `pytest tests`.

## Made with ImageGoNord
This could not have been possible without the [ImageGoNord](https://github.com/Schroedinger-Hat/ImageGoNord-Web) project by
[Schrödinger Hat](https://github.com/Schroedinger-Hat). You can sponsor it at https://opencollective.com/schrodinger-hat.

## Thanks to all the contributors of the original collection :)

<a href = "https://github.com/linuxdotexe/nordic-wallpapers/graphs/contributors">

  <img src = "https://contrib.rocks/image?repo=linuxdotexe/nordic-wallpapers"/>

</a>
