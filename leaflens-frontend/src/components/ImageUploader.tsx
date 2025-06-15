// src/components/ImageUploader.tsx
'use client';

import { FC, ChangeEvent, useState } from 'react';

interface ImageUploaderProps {
  onSelect: (file: File) => void;
}

export const ImageUploader: FC<ImageUploaderProps> = ({ onSelect }) => {
  const [preview, setPreview] = useState<string | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    if (!file) {
      setPreview(null);
      return;
    }

    // create a base64 preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result as string);
      onSelect(file);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="flex flex-col items-center">
      <input
        type="file"
        accept="image/*"
        capture="environment"        // hints to open camera on mobile
        onChange={handleFileChange}
        className="mb-4"
      />
      {preview && (
        <img
          src={preview}
          alt="preview"
          className="w-64 h-auto rounded-lg shadow"
        />
      )}
    </div>
  );
};
