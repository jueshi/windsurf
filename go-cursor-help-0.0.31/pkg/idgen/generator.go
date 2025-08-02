package idgen

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"
)

// Generator handles secure ID generation for machines and devices
type Generator struct{}

// NewGenerator creates a new ID generator
func NewGenerator() *Generator {
	return &Generator{}
}

// Helper methods
// -------------

// simulateWork adds a small delay to make progress visible
func (g *Generator) simulateWork() {
	time.Sleep(800 * time.Millisecond)
}

// generateRandomHex generates a random hex string of specified length
func (g *Generator) generateRandomHex(length int) (string, error) {
	bytes := make([]byte, length)
	if _, err := rand.Read(bytes); err != nil {
		return "", fmt.Errorf("failed to generate random bytes: %w", err)
	}
	return hex.EncodeToString(bytes), nil
}

// Public methods
// -------------

// GenerateMachineID generates a new 32-byte machine ID
func (g *Generator) GenerateMachineID() (string, error) {
	g.simulateWork()
	return g.generateRandomHex(32)
}

// GenerateMacMachineID generates a new 64-byte MAC machine ID
func (g *Generator) GenerateMacMachineID() (string, error) {
	g.simulateWork()
	return g.generateRandomHex(64)
}

// GenerateDeviceID generates a new device ID in UUID format
func (g *Generator) GenerateDeviceID() (string, error) {
	g.simulateWork()
	id, err := g.generateRandomHex(16)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s-%s-%s-%s-%s",
		id[0:8], id[8:12], id[12:16], id[16:20], id[20:32]), nil
}
