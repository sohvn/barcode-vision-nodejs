export type BarcodeType =
  | "Barcode"
  | "QRCode";

export type ScanResult = {
  type: BarcodeType;
  confidence: number;
  value: string | null;
  decoded: boolean;
};