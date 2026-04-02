import os
from dotenv import load_dotenv
from supadata import Supadata

# Load environment variables from .env file
load_dotenv()

def fetch_and_save_transcript(url, output_file="my_video_transcript.txt"):
    """
    Fetches the transcript for a given URL using the Supadata API and saves it to a text file.
    """
    # Retrieve API key from environment variables
    api_key = os.getenv("SUPADATA_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        print("Error: Valid SUPADATA_API_KEY not found in .env file.")
        print("Please update your .env file with your actual Supadata API key.")
        return

    print("Initializing Supadata client...")
    try:
        supadata = Supadata(api_key=api_key)
        
        print(f"Fetching transcript for URL: {url}")
        def format_timestamp(ms):
            seconds = ms // 1000
            minutes = seconds // 60
            hours = minutes // 60
            minutes = minutes % 60
            seconds = seconds % 60
            if hours > 0:
                return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
            else:
                return f"[{minutes:02d}:{seconds:02d}]"

        # Fetch the transcript structured data
        try:
            transcript = supadata.transcript(url=url)
            
            # Save to the specified output file
            with open(output_file, 'w', encoding='utf-8') as f:
                if hasattr(transcript, 'content'):
                    for chunk in transcript.content:
                        if hasattr(chunk, 'offset') and hasattr(chunk, 'text'):
                            timestamp = format_timestamp(getattr(chunk, 'offset'))
                            text = getattr(chunk, 'text')
                            f.write(f"{timestamp} {text}\n")
                        else:
                            f.write(str(chunk) + "\n")
                elif isinstance(transcript, str):
                    f.write(transcript)
                else:
                    f.write(str(transcript))
                    
            print(f"Success! Transcript saved to {output_file}")
            
        except AttributeError:
             print("Error: Make sure you have installed the correct supadata package.")
             
    except Exception as e:
         print(f"An error occurred: {e}")

if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    if url.strip():
        fetch_and_save_transcript(url.strip())
    else:
        print("No URL provided. Exiting.")
