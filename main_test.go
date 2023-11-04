package main

import (
	"fmt"
	"log"
	"testing"
)

func TestXOr(t *testing.T) {
	arch := []int{2, 3, 2, 1}
	input := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	output := [][]float64{
		{1},
		{0},
		{0},
		{1},
	}

	nn, err := NewNN(arch)
	if err != nil {
		log.Fatal(err)
	}
	nn.InitWeights()

	nn.FitFiniteDiffs(input, output)
	for _, in := range input {
		prediction, _ := nn.FeedForeward(in)
		fmt.Println(in, prediction)
	}
}

func TestDumb(t *testing.T) {
	nn := NN{}
	nn.Layers = []DenseLayer{
		{
			Ins:  2,
			Outs: 2,
			Weights: [][]float64{
				{-3, 2}, {1, 6},
			},
			Biases: make([]float64, 2),
			Output: make([]float64, 2),
		},
		{
			Ins:  2,
			Outs: 1,
			Weights: [][]float64{
				{8},
				{4},
			},
			Biases: make([]float64, 1),
			Output: make([]float64, 1),
		},
	}

	nn.FeedForeward([]float64{2, 6})
}
