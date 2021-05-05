install:
	python3 -m venv .env && source .env/bin/activate
	python3 -m pip install -r requirements.txt

cleanup:
	rm -r model_backup || exit 0
	mkdir model_backup && mv model model_backup/ || exit 0
	rm -r model || exit 0
	rm -r model_folds || exit 0

# custom model
run: cleanup	
	python3 train.py --epochs=125 --parallel_folds

# transfer learning example
run-transfer: cleanup		
	python3 train_transfer.py --epochs=125 --parallel_folds

# extract features mean
extract-features:
	python3 extract_features.py
extract-demo:
	python3 face_identify_demo.py


test:
	python3 test_model.py --image_path="test_family_images/test4.jpg"

prepare-data:
	python3 face_detection_operation.py "dataset-family-collected" "dataset-family-small"
