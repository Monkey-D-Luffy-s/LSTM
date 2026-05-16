# TODO - Improve LSTM Predictor Accuracy

- [x] Update `prepare_train_data` to make horizon/label alignment explicit (t -> t+horizon).

- [x] Add validation split during training.

- [x] Compute best classification threshold on validation set; persist it (e.g., `threshold.json`).

- [x] Update `predict()` to generate horizon-aligned rolling predictions across the provided 200 candles (instead of single-step).

- [x] Save updated model + scaler + threshold.

- [ ] Provide a command to train and run prediction for quick verification.

