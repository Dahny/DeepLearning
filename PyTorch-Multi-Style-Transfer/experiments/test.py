import os
import argparse

from main import evaluate

def benchmark():
	style_directory = "../../Benchmark/style"
	content_directory = "../../Benchmark/content"

	for content_file in os.listdir(os.fsencode(content_directory)):
			content_image = os.fsdecode(content_file)
			content_path = os.path.join(content_directory, content_image)
			print(content_image)
			# print(content_path)

			for style_file in os.listdir(os.fsencode(style_directory)):
					style_image = os.fsdecode(style_file)
					style_path = os.path.join(style_directory, style_image)
					# print(style_image)
					print(style_path)

					output_image = content_image.split(".")[0] + "-" +style_image.split(".")[0] + ".jpg"
					output_path = os.path.join("images/test", output_image)

					parser = argparse.ArgumentParser()
					parser.add_argument('--content-image')
					parser.add_argument('--style-size', type=int, default=512)
					parser.add_argument('--ngf', type=int, default=128)
					parser.add_argument('--style-image')
					parser.add_argument('--output-image')
					parser.add_argument('--model')
					parser.add_argument('--content-size', type=int)
					parser.add_argument('--cuda', type=int)
					args = parser.parse_args(['--content-image', content_path,
																		'--style-image', style_path,
																		'--output-image', output_path,
																		'--model', "models/21styles.model",
																		'--content-size', '256',
																		'--cuda', '0'])
					evaluate(args)


benchmark()