from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def main():
    x_train_data = pd.read_csv("./normalised_train_set/x_train_normalised_with_y_train_smpl.csv", delimiter=",")

    x = x_train_data.iloc[:, 0:1600]
    y = x_train_data.iloc[:, -1]
    y = y.astype(int)

    # Create a model and fit it
    model = LinearRegression().fit(x, y)

    # Get results
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    # Predict response
    y_pred = model.predict(x)
    y_pred = y_pred.astype(int)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
    # print('predicted response:', y_pred, sep='\n')


if __name__ == "__main__":
    main()