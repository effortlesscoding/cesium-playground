// Synthetic terrain for testing ground clamping offline: smooth hills a few hundred meters high
// and a couple of kilometers wide, so fields at a fixed height visibly float or sink.
import { CustomHeightmapTerrainProvider, GeographicTilingScheme, Math as CesiumMath } from "cesium";

const SIZE = 65;
const AMPLITUDE_M = 150;
const WAVELENGTH_DEG = 0.03;

function heightAt(lonDeg, latDeg) {
  const k = (2 * Math.PI) / WAVELENGTH_DEG;
  return AMPLITUDE_M * (0.5 + 0.5 * Math.sin(lonDeg * k) * Math.cos(latDeg * k));
}

export function hillsTerrain() {
  const tilingScheme = new GeographicTilingScheme();
  return new CustomHeightmapTerrainProvider({
    width: SIZE,
    height: SIZE,
    tilingScheme,
    callback: (x, y, level) => {
      const r = tilingScheme.tileXYToRectangle(x, y, level);
      const west = CesiumMath.toDegrees(r.west);
      const east = CesiumMath.toDegrees(r.east);
      const north = CesiumMath.toDegrees(r.north);
      const south = CesiumMath.toDegrees(r.south);
      const heights = new Float32Array(SIZE * SIZE);
      // Rows run north to south.
      for (let j = 0; j < SIZE; j++) {
        const lat = north + ((south - north) * j) / (SIZE - 1);
        for (let i = 0; i < SIZE; i++) {
          heights[j * SIZE + i] = heightAt(west + ((east - west) * i) / (SIZE - 1), lat);
        }
      }
      return heights;
    },
  });
}
