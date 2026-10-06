from dicom_utils import load_dicom_image
from PIL import Image


dicom_path = r"C:\Users\ravis\Desktop\mammogram_data\manifest-1791038360788\cbis_ddsm\P_00038_LEFT_CC.dcm\96009\1-63992\9a3af421-bb04-461c-9445-8f26f9ab34e7.dcm"

image = load_dicom_image(dicom_path)

print("Preprocessing successful")
print("Image shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum pixel value:", image.min())
print("Maximum pixel value:", image.max())


output_path = r"C:\Users\ravis\Desktop\mammogram_data\mammogram_test.png"

Image.fromarray(image).save(output_path)

print("PNG saved to:", output_path)