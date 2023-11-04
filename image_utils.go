package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
)

var screenChars = []string{"  ", "..", ",,", "oo", "OO", "00", "##"}

func ReadPNG(path string) (image.Image, error) {
	// Open the file
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Decode the file into an image
	img, err := png.Decode(file)
	if err != nil {
		return nil, err
	}

	return img, nil
}

func PrintImage(img image.Image) {
	width, height := img.Bounds().Size().X, img.Bounds().Size().Y

	for y := 0; y < width; y++ {
		for x := 0; x < height; x++ {
			fmt.Print(toScreenChar(img.At(x, y)))
		}
		fmt.Println()
	}
}

func toScreenChar(color color.Color) string {
	intensity, _, _, _ := color.RGBA()
	intensity = intensity / 256

	colorStepInterval := 256 / float32(len(screenChars))

	return screenChars[uint32(float32(intensity)/colorStepInterval)]
}
