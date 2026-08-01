import os
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

DATASET_1_URLS = [
    "https://raw.githubusercontent.com/sumana-2705/Predicting-Student-Performance-Index/main/Student_Performance.csv",
    "https://raw.githubusercontent.com/mohammadtalalai/Student-Performance-Prediction/main/Student_Performance.csv"
]

DATASET_2_URLS = [
    "https://raw.githubusercontent.com/khushbupoul/student-performance-app-using-machine-learning/master/StudentPerformanceFactors.csv",
    "https://raw.githubusercontent.com/khushbupoul/student-performance-app-using-machine-learning/main/StudentPerformanceFactors.csv",
    "https://raw.githubusercontent.com/ArianJr/student-performance-deep-learning/main/StudentPerformanceFactors.csv",
    "https://raw.githubusercontent.com/ArianJr/student-performance-deep-learning/master/StudentPerformanceFactors.csv",
    "https://raw.githubusercontent.com/Ayushkumar418/Student_Performance_Predictor/main/StudentPerformanceFactors.csv",
    "https://raw.githubusercontent.com/Ayushkumar418/Student_Performance_Predictor/master/StudentPerformanceFactors.csv",
    "https://raw.githubusercontent.com/Shivi2599/Student_Performance_Factors_Kaggle/main/StudentPerformanceFactors.csv"
]

def download_file(urls, filename):
    filepath = os.path.join(DATA_DIR, filename)
    for url in urls:
        print(f"Trying to download {url}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    content = response.read()
                    # Check if the content is actual data, not a short error message
                    if len(content) > 1000:
                        with open(filepath, 'wb') as f:
                            f.write(content)
                        print(f"Successfully downloaded {filename} from {url} ({len(content)} bytes).")
                        return True
                    else:
                        print(f"URL responded, but content too short ({len(content)} bytes). Skipping.")
        except urllib.error.URLError as e:
            print(f"Failed to fetch from {url}: {e}")
    print(f"Error: All URLs failed for {filename}")
    return False

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}")
        
    download_file(DATASET_1_URLS, "Student_Performance.csv")
    download_file(DATASET_2_URLS, "StudentPerformanceFactors.csv")

if __name__ == "__main__":
    main()
