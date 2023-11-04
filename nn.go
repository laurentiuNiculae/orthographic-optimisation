package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func Linear(x float64) float64 {
	return x
}

type DenseLayer struct {
	Ins     int
	Outs    int
	Weights [][]float64
	Biases  []float64

	// This will keep the output of the layer
	Output []float64

	WeightsUpdate [][]float64
}

func NewDenseLayer(in, out int) DenseLayer {
	weights := make([][]float64, in)

	for i := 0; i < in; i++ {
		weights[i] = make([]float64, out)
	}

	weightsUpdates := make([][]float64, in)

	for i := 0; i < in; i++ {
		weightsUpdates[i] = make([]float64, out)
	}

	return DenseLayer{
		Ins:           in,
		Outs:          out,
		Weights:       weights,
		Biases:        make([]float64, out),
		Output:        make([]float64, out),
		WeightsUpdate: weightsUpdates,
	}
}

func (ld *DenseLayer) Feed(input []float64) error {
	if len(input) != ld.Ins {
		return fmt.Errorf("input size '%d' doesn't match '%d'", len(input), len(ld.Weights))
	}

	for out := 0; out < ld.Outs; out++ {
		var sum float64
		for in := 0; in < ld.Ins; in++ {
			sum += input[in] * ld.Weights[in][out]
		}

		ld.Output[out] = Sigmoid(sum + ld.Biases[out])
	}

	return nil
}

type NN struct {
	Layers []DenseLayer
}

func NewNN(arch []int) (NN, error) {
	nn := NN{}

	if len(arch) < 3 {
		return NN{}, errors.New("Need at least 3 inputs in the arch. 1 inputs, 1 hidden layer, 1 output shape")
	}

	for i := 0; i < len(arch)-1; i++ {
		nn.Layers = append(nn.Layers, NewDenseLayer(arch[i], arch[i+1]))
	}

	return nn, nil
}

func (nn *NN) InitWeights() {
	for l := range nn.Layers {
		for x := range nn.Layers[l].Weights {
			for y := range nn.Layers[l].Weights[0] {
				nn.Layers[l].Weights[x][y] = rand.Float64()
			}
		}
	}
}

func (nn *NN) FeedForeward(input []float64) ([]float64, error) {
	in := input
	for i := range nn.Layers {
		err := nn.Layers[i].Feed(in)
		if err != nil {
			return []float64{}, err
		}

		in = nn.Layers[i].Output
	}

	return in, nil
}

const learningRate = 0.1

func (nn *NN) FitFiniteDiffs(input [][]float64, out [][]float64) {
	for epoch := 0; epoch < 100000; epoch++ {
		for i := 0; i < len(input); i++ {
			var h float64 = 0.0001

			// go through all layers parameters
			for l := 0; l < len(nn.Layers); l++ {
				// weights
				for x := 0; x < nn.Layers[l].Ins; x++ {
					for y := 0; y < nn.Layers[l].Outs; y++ {
						prediction1, _ := nn.FeedForeward(input[i])
						cost1 := MeanSquareError(prediction1, out[i])

						// tweack a little bit 1 parameter by adding h
						oldVal := nn.Layers[l].Weights[x][y]
						nn.Layers[l].Weights[x][y] += h
						prediction2, _ := nn.FeedForeward(input[i])
						cost2 := MeanSquareError(prediction2, out[i])
						nn.Layers[l].Weights[x][y] = oldVal

						// derivative for the parameter
						derivative := (cost2 - cost1) / h
						nn.Layers[l].WeightsUpdate[x][y] = -derivative * learningRate
					}
				}
			}

			// update parameters
			for l := 0; l < len(nn.Layers); l++ {
				// weights
				for x := 0; x < nn.Layers[l].Ins; x++ {
					for y := 0; y < nn.Layers[l].Outs; y++ {
						nn.Layers[l].Weights[x][y] += nn.Layers[l].WeightsUpdate[x][y]
					}
				}
			}
		}

		if epoch%500 == 0 {
			totalCost := float64(0)
			for i := 0; i < len(input); i++ {
				prediction, _ := nn.FeedForeward(input[i])
				totalCost += MeanSquareError(prediction, out[i])
			}
			fmt.Println(totalCost)
		}
	}
}

func MeanSquareError(expected, actual []float64) float64 {
	var sum float64

	for i := range expected {
		diff := (expected[i] - actual[i])
		sum += diff * diff
	}

	return sum / float64(len(expected))
}
