import joblib
import numpy as np
m = joblib.load("models/model.pkl")
fn = getattr(m, "feature_names_in_", None)
print("len(feature_names_in_):", len(fn))
print(fn)

# Try to inspect feature importance / coefficients
if hasattr(m, "feature_importances_"):
    print("Feature importances:", list(zip(fn, m.feature_importances_)))
elif hasattr(m, "coef_"):
    # flatten if multiclass
    print("Coefficients shape:", m.coef_.shape)
    for i, coef_row in enumerate(m.coef_):
        print(f"class {i} coefs (top 10 by abs):")
        idx = np.argsort(np.abs(coef_row))[::-1][:10]
        for j in idx:
            print(fn[j], coef_row[j])
else:
    print("Model has no straightforward coef_/feature_importances_. Consider permutation importance.")
