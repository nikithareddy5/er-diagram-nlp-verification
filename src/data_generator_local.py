import pandas as pd
import re
import os
import requests
import random
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io
import json


import os

OPENAI_API_KEY_COURSE = os.getenv("OPENAI_API_KEY_COURSE")
if not OPENAI_API_KEY_COURSE:
    raise RuntimeError("OPENAI_API_KEY_COURSE not SET in environment variables")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5.1"


# Input/Output Paths
INPUT_CSV = "kaggle_data/system_requirements_5000.csv"
OUTPUT_DIR = "data"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
IMAGE_DIR_A = os.path.join(IMAGE_DIR, "dataset_A_PERFECT")
IMAGE_DIR_B = os.path.join(IMAGE_DIR, "dataset_B_SYNONYM")
IMAGE_DIR_C = os.path.join(IMAGE_DIR, "dataset_C_LOGIC_ERROR")

# Dataset Configuration
NUM_SAMPLES = 20  # Number of rows to use for each dataset
DEBUG_PRINT_COUNT = 2  # Print detailed info for the first N samples


total_input_tokens = 0
total_output_tokens = 0

# Create Directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR_A, exist_ok=True)
os.makedirs(IMAGE_DIR_B, exist_ok=True)
os.makedirs(IMAGE_DIR_C, exist_ok=True)


def call_llm_model(prompt, temperature=0.2, max_tokens=2000):

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY_COURSE}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        global total_input_tokens, total_output_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        return content, input_tokens, output_tokens

    except requests.exceptions.Timeout:
        print("   OpenAI API timeout (2 minutes)")
        return None, 0, 0
    except requests.exceptions.HTTPError as e:
        print(f"   OpenAI API HTTP error: {e}")
        if hasattr(e.response, "text"):
            print(f"   Details: {e.response.text}")
        return None, 0, 0
    except Exception as e:
        print(f"   OpenAI API call failed: {e}")
        return None, 0, 0
    


def extract_plantuml(text):
    if not text:
        return None
    
    match = re.search(r'(@startuml.*?@enduml)', text, re.DOTALL)
    if match:
        return match.group(1)
    
    match = re.search(r'```(?:plantuml)?(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "@startuml" not in code:
            code = "@startuml\n" + code
        if "@enduml" not in code:
            code = code + "\n@enduml"
        return code
    
    return text

def validate_plantuml(code):
    if not code:
        return False, "Code is empty"

    if "@startuml" not in code:
        return False, "Missing @startuml"
    if "@enduml" not in code:
        return False, "Missing @enduml"
    
    lines = code.split('\n')
    has_class = any('class ' in line for line in lines)
    has_entity = any('entity ' in line for line in lines)
    has_braces = any('{' in line or '}' in line for line in lines)
    
    if not (has_class or has_entity or has_braces):
        return False, "No entity definition found"

    open_braces = code.count('{')
    close_braces = code.count('}')
    if open_braces != close_braces:
        return False, f"Braces mismatch"

    return True, "Syntax correct"

def generate_image(plantuml_code, filename, error_type="none", requirement_original="", requirement_changed="", image_dir=IMAGE_DIR):
    if not plantuml_code:
        return None
    
    save_path = os.path.join(image_dir, filename)
    
    try:
        
        # Try to get diagram image from service
        diagram_img = None
        response = None
        raw_png_data = None
        
        # Method 1: Try kroki.io with POST plain text (text/plain)
        try:
            url = "https://kroki.io/plantuml/png"
            headers = {'Content-Type': 'text/plain; charset=utf-8'}
            response = requests.post(url, data=plantuml_code.encode('utf-8'), headers=headers, timeout=30)
            
            if response.status_code == 200 and response.content[:4] == b'\x89PNG':
                raw_png_data = response.content
                print(f"   Success: Image generated via Method 1 (kroki.io POST) - {len(raw_png_data)} bytes")
        except Exception as e1:
            print(f"   Method 1 (kroki.io POST) failed: {str(e1)}")
        
        # Method 4: Try kroki.io with POST JSON (application/json) - fallback
        if raw_png_data is None:
            try:
                url = "https://kroki.io/plantuml/png"
                headers = {'Content-Type': 'application/json'}
                payload = json.dumps({"diagram_source": plantuml_code})
                response = requests.post(url, data=payload, headers=headers, timeout=30)
                
                if response.status_code == 200 and response.content[:4] == b'\x89PNG':
                    raw_png_data = response.content
                    print(f"   Success: Image generated via Method 4 (kroki.io JSON) - {len(raw_png_data)} bytes")
            except Exception as e4:
                print(f"   Method 4 (kroki.io JSON) failed: {str(e4)}")
        
        # If we got raw PNG data, save it first to a temp location for debugging
        if raw_png_data is not None:
            # Create a raw images subdirectory for debugging
            raw_dir = os.path.join(image_dir, "_raw")
            os.makedirs(raw_dir, exist_ok=True)
            raw_save_path = os.path.join(raw_dir, filename)
            
            # Save raw PNG first
            try:
                with open(raw_save_path, 'wb') as f:
                    f.write(raw_png_data)
                print(f"   Success: Saved raw PNG: {raw_save_path}")
            except Exception as e:
                print(f"   Error saving raw PNG: {e}")
            
            # Now try to open and process the image
            try:
                diagram_img = Image.open(io.BytesIO(raw_png_data))
                diagram_img.load()  # Force load
                print(f"   Success: Image loaded successfully: {diagram_img.width}x{diagram_img.height}")
            except Exception as e:
                print(f"   Error loading image from PNG data: {e}")
                diagram_img = None
        
        # Additional dimension validation
        if diagram_img is not None:
            img_width = diagram_img.width if hasattr(diagram_img, 'width') else 0
            img_height = diagram_img.height if hasattr(diagram_img, 'height') else 0
            
            print(f"   Image dimensions: {img_width}x{img_height}")
            
            if img_width <= 0 or img_height <= 0:
                print(f"   Warning: Invalid image dimensions {img_width}x{img_height}, using placeholder")
                diagram_img = None
        
        # If we have a valid diagram image, add metadata
        if diagram_img is not None:
            try:
                img_width = diagram_img.width
                img_height = diagram_img.height
                
                # Calculate metadata area height based on content
                line_height = 15  # Increase line height to avoid text overlap
                metadata_lines = [
                    f"Error Type: {error_type}",
                    f"Label: {'MATCH' if error_type == 'none' else 'MATCH' if error_type == 'synonym' else 'MISMATCH'}",
                    "",
                    f"Original: {requirement_original}",
                ]
                
                if error_type != "none" and requirement_changed:
                    metadata_lines.append(f"Changed: {requirement_changed}")
                
                metadata_lines.extend([
                    "",
                    f"PlantUML Code (sample):"
                ])
                
                # Add first few lines of PlantUML (limited to prevent overflow)
                plantuml_lines = plantuml_code.split('\n')  # Reduce number of lines
                for line in plantuml_lines:
                    if line.strip():
                        metadata_lines.append(f"  {line}")
                
                # Calculate total metadata height
                metadata_height = max(len(metadata_lines) * line_height + 40, 180)
                
                # Create canvas with safe dimensions
                canvas_width = max(img_width, 800)
                canvas_height = img_height + metadata_height
                
                print(f"   Canvas size: {canvas_width}x{canvas_height}")
                
                # Create a new image with diagram on top and metadata below
                new_img = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))
                
                # Paste the diagram at the top
                new_img.paste(diagram_img, (0, 0))
                print(f"   Success: Pasted diagram at (0, 0)")
                
                # Add metadata text below
                draw = ImageDraw.Draw(new_img)
                
                # Try to load fonts with better error handling
                font_text = None
                font_paths = [
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/System/Library/Fonts/Arial.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
                ]
                
                for font_path in font_paths:
                    try:
                        font_text = ImageFont.truetype(font_path, 10)
                        print(f"   Success: Loaded font: {font_path}")
                        break
                    except:
                        continue
                
                # If no TrueType font found, skip text rendering to avoid errors
                if font_text is None:
                    print(f"   Warning: No TrueType font available, saving image without metadata text")
                    # Just save the diagram without text metadata
                    new_img.save(save_path)
                    print(f"   Success: Saved image (without text metadata): {save_path}")
                    return save_path
                
                # Draw text with the loaded font
                y_pos = img_height + 15
                
                for line in metadata_lines:
                    try:
                        if line and y_pos < canvas_height - 25:
                            draw.text((10, y_pos), line, fill=(0, 0, 0), font=font_text)
                            y_pos += line_height
                        elif not line:
                            y_pos += 5  # Smaller spacing for empty lines
                    except Exception as e:
                        print(f"   Warning: Failed to draw line '{line}': {e}")
                        continue
                
                print(f"   Success: Added metadata text")
                
                # Save the new image
                new_img.save(save_path)
                print(f"   Success: Saved composite image: {save_path}")
                return save_path
                
            except Exception as e:
                print(f"   Error processing image with metadata: {type(e).__name__}: {str(e)}")
                # If metadata addition fails, just save the raw image
                if raw_png_data is not None:
                    try:
                        with open(save_path, 'wb') as f:
                            f.write(raw_png_data)
                        print(f"   Success: Saved raw image (without metadata): {save_path}")
                        return save_path
                    except Exception as e2:
                        print(f"   Error saving raw image: {e2}")
        
        # If we couldn't get a diagram, create placeholder
        return _create_placeholder_image(save_path, plantuml_code, "Both kroki.io methods failed")
            
    except Exception as e:
        print(f"   Error generating image: {type(e).__name__}: {str(e)}")
        return _create_placeholder_image(save_path, plantuml_code, str(e))


def _create_placeholder_image(save_path, plantuml_code, error_msg):
    try:
        placeholder = Image.new('RGB', (900, 450), color=(255, 220, 220))
        draw = ImageDraw.Draw(placeholder)
        
        # Try to load fonts with better fallback
        font_title = None
        font_text = None
        
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]
        
        for font_path in font_paths:
            try:
                font_title = ImageFont.truetype(font_path, 14)
                font_text = ImageFont.truetype(font_path, 10)
                break
            except:
                continue
        
        # If no font loaded, create a simple placeholder without text
        if font_title is None or font_text is None:
            # Draw a colored border to indicate error
            draw.rectangle([(0, 0), (899, 449)], outline=(200, 0, 0), width=5)
            draw.rectangle([(10, 10), (889, 439)], outline=(150, 0, 0), width=2)
            placeholder.save(save_path)
            print(f"   Saved minimal placeholder image: {save_path}")
            return save_path
        
        # Draw text if fonts are available
        try:
            draw.text((20, 20), "Warning: Diagram Generation Failed", fill=(200, 0, 0), font=font_title)
            draw.text((20, 50), f"Error: {error_msg}", fill=(150, 0, 0), font=font_text)
            draw.text((20, 75), "PlantUML Code Provided", fill=(100, 100, 100), font=font_text)
            
            # Show first few lines of PlantUML code
            lines = plantuml_code.split('\n')
            y = 105
            for line in lines:
                if line.strip():
                    draw.text((30, y), line, fill=(50, 50, 50), font=font_text)
                    y += 20
            
            draw.text((20, 400), "A placeholder image was saved. Check the code or service status.", 
                      fill=(100, 100, 100), font=font_text)
        except Exception as e:
            print(f"   Warning: Could not add text to placeholder: {e}")
        
        placeholder.save(save_path)
        print(f"   Saved placeholder image: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"   Failed to create placeholder image: {e}")
        return None


def generate_base_plantuml(requirement_text, debug_print=False):
    prompt = f"""
You are an expert business analyst and data modeler.
Your task is to transform ONE natural-language requirement sentence into a PlantUML ER diagram using Chen's notation.

Rules:
- Keep the ERD as simple as possible and close to the original sentence.
- Use ONLY clear nouns from the requirement sentence as entities.
- Use the main verbs/actions in the sentence as relationship names (in UPPER_SNAKE_CASE). You may include
  important adjectives that change the meaning of the action, e.g. ADD_NEW_BOOK, REMOVE_DAMAGED_BOOK.
- You may add exactly ONE id attribute per entity (e.g. StaffId, BookId, CustomerId).
- You may add attributes ONLY if they are explicit NOUNS or NOUN PHRASES
  (e.g. "student performance", "grades", "attendance", "images"),
  OR if you need a simple state/condition attribute to distinguish cases where adjectives modify a noun
  (e.g. "new books" vs "damaged ones") — then you may add a generic attribute such as Status or Condition
  on that entity.
- Do NOT create attributes from pure adjectives/adverbs or frequency words such as
  "monthly", "daily", "weekly", "automatically", "quickly", etc.
  These should be treated as business rules, NOT as attributes in the ER diagram.
- If the requirement says "about X" or "including X", then X should be modeled as an attribute
  on a suitable entity (for example, Performance on NOTIFICATION or STUDENT).
- Every entity you create MUST participate in at least one relationship. Do not leave any entity isolated.
- When a noun modifies another noun (e.g. "customer shipping information", "customer billing information"),
  you must connect those entities with a relationship (for example, STORE_CUSTOMER_INFORMATION connecting
  SYSTEM, CUSTOMER, SHIPPING_INFORMATION, BILLING_INFORMATION), instead of leaving CUSTOMER unconnected.
- Do NOT invent extra entities or attributes that are not clearly present in the requirement.

Example
Requirement:
Staff shall be able to add new books and remove damaged ones.

PlantUML ERD:

@startchen
entity STAFF {{
  StaffId <<key>>
}}

entity BOOK {{
  BookId <<key>>
  Condition
}}

relationship ADD_NEW_BOOK {{
}}

relationship REMOVE_DAMAGED_BOOK {{
}}

ADD_NEW_BOOK =1= STAFF
ADD_NEW_BOOK -N- BOOK

REMOVE_DAMAGED_BOOK =1= STAFF
REMOVE_DAMAGED_BOOK -N- BOOK
@endchen

NOW DO THE SAME FOR THIS INPUT

Transform the following requirement into a PlantUML ER diagram using Chen's notation, following the EXACT style of the example above.

REQUIREMENT TEXT:
{requirement_text}
"""
    
    if debug_print:
        print("\n" + "─"*60)
        print("Prompt:")
        print("─"*60)
        print(prompt)
        print("─"*60)
    
    start_time = time.time()
    content, in_tokens, out_tokens = call_llm_model(prompt, temperature=0.2, max_tokens=2000)
    api_time = time.time() - start_time

    result = extract_plantuml(content)

    for attempt in range(3):
        is_valid, msg = validate_plantuml(result)
        if is_valid:
            if debug_print:
                print(f"API call time: {api_time:.2f}s")
                print(f"Input: {in_tokens:,} tokens | Output: {out_tokens:,} tokens")
                print(f"Total so far: (Input: {total_input_tokens:,} | Output: {total_output_tokens:,})")
                print("\n" + "─"*60)
                print("Generated PlantUML:")
                print("─"*60)
                print(result)
                print("─"*60 + "\n")
            break
        else:
            if attempt < 2:
                prompt += f"\n\nPREVIOUS ERROR: {msg}\nPlease fix and regenerate."
                content, in_tokens, out_tokens = call_llm_model(prompt, temperature=0.0, max_tokens=2000)
                result = extract_plantuml(content)

    return result, api_time

def change_requirement_synonym(requirement_text, debug_print=False):
    prompt = f"""Given this requirement text:
{requirement_text}

Task: Replace 2-4 key terms (entity names, attribute names, or domain/role terms) with MEANINGFUL SYNONYMS.

IMPORTANT: Make SUBSTANTIAL and VARIED replacements, not minor ones. Examples of good replacements:
- Staff→Manager, Staff→Personnel, Staff→Employee, Staff→Operator
- Doctor→Physician, Doctor→Practitioner, Doctor→Clinician, Doctor→Specialist
- Book→Volume, Book→Publication, Book→Work, Book→Item
- Patient→Client, Patient→Individual, Patient→Person, Patient→Recipient
- Hospital→Medical Center, Hospital→Clinic, Hospital→Facility, Hospital→Institution
- Add→Insert, Add→Register, Add→Create, Add→Submit
- Email→E-mail address, Email→Electronic mail, Email→Contact info
- Schedule→Arrange, Schedule→Plan, Schedule→Book, Schedule→Assign

Rules:
1. REPLACE 2-4 terms with CLEAR, MEANINGFUL SYNONYMS (not trivial variations)
2. The core meaning must stay the SAME (genuine synonyms/close alternatives)
3. Choose terms that would be found in real domain documents
4. Make replacements that are NOTICEABLY DIFFERENT, not just minor variations
5. Keep the requirement structure and grammar intact
6. Output ONLY the modified requirement text (no explanations, no commentary)
7. Maintain all other details exactly as they are
"""
    
    if debug_print:
        print("\n" + "─"*60)
        print("Synonym Change Prompt (Requirement Level):")
        print("─"*60)
        print(prompt)
        print("─"*60)
    
    start_time = time.time()
    content, in_tokens, out_tokens = call_llm_model(prompt, temperature=0.5, max_tokens=2000)
    api_time = time.time() - start_time

    result = content.strip()

    if debug_print:
        print(f"API call time: {api_time:.2f}s")
        print(f"Input: {in_tokens:,} tokens | Output: {out_tokens:,} tokens")
        print(f"Total so far: (Input: {total_input_tokens:,} | Output: {total_output_tokens:,})")
        print("\n" + "─"*60)
        print("After synonym change (requirement):")
        print("─"*60)
        print(result)
        print("─"*60 + "\n")

    return result, api_time

def change_requirement_logic_error(requirement_text, debug_print=False):
    """Apply logic error changes to Requirement text level."""
    prompt = f"""Given this requirement text:
{requirement_text}

Task: Introduce 2-3 CLEAR LOGIC ERRORS that make the requirement obviously WRONG.

Error examples (choose different types):
1. Reverse WHO does WHAT: "teachers instruct students" → "students instruct teachers"
2. Change the ACTION to something illogical: "author writes book" → "book writes author"
3. Flip ONE/MANY incorrectly: "one course has many students" → "one student has many courses" (when illogical)
4. Reverse the direction: "employee reports to manager" → "manager reports to employee"
5. Change verb to opposite meaning: "doctor treats patient" → "patient treats doctor"

CRITICAL: The errors must be OBVIOUSLY WRONG when you read them. Someone should immediately notice "that doesn't make sense!"

Requirements:
- Make 2-3 changes minimum
- Each change should make the logic clearly INCORRECT
- Keep similar sentence length
- The modified text should sound wrong/illogical
- Output ONLY the modified requirement text (no explanations)

Example:
Original: "The system shall allow teachers to assign homework to students."
Wrong: "The system shall allow students to assign homework to teachers."
(This is CLEARLY backwards!)
"""
    
    if debug_print:
        print("\n" + "─"*60)
        print("Logic Error Change Prompt (Requirement Level):")
        print("─"*60)
        print(prompt)
        print("─"*60)
    
    start_time = time.time()
    content, in_tokens, out_tokens = call_llm_model(prompt, temperature=0.7, max_tokens=2000)  
    api_time = time.time() - start_time

    result = content.strip()

    if debug_print:
        print(f"API call time: {api_time:.2f}s")
        print(f"Input: {in_tokens:,} tokens | Output: {out_tokens:,} tokens")
        print(f"Total so far: (Input: {total_input_tokens:,} | Output: {total_output_tokens:,})")
        print("\n" + "─"*60)
        print("After logic error change (requirement):")
        print("─"*60)
        print(result)
        print("─"*60 + "\n")

    return result, api_time

def generate_dataset_a(texts):
    print(f"\n{'='*40} Generating Dataset A: Perfect Match {'='*40}")
    print(f"Using first {len(texts)} rows of data")
    print(f"Label: MATCH (perfect match)")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    rows = []
    total_time = 0
    
    for i, text in enumerate(texts):
        sample_id = f"dataset_A_PERFECT_{i+1:04d}"
        debug_print = (i < DEBUG_PRINT_COUNT)

        if debug_print:
            print(f"\n{'='*25} Sample {i+1}/{len(texts)} - {sample_id} {'='*25}")
            print(f"Requirement: {text}")
        
        sample_start = time.time()
        plantuml_code, api_time = generate_base_plantuml(text, debug_print=debug_print)
        img_path = generate_image(plantuml_code, f"{sample_id}.png", error_type="none", requirement_original=text, requirement_changed="", image_dir=IMAGE_DIR_A)
        sample_time = time.time() - sample_start
        total_time += sample_time
        
        if debug_print:
            print(f"Sample total time: {sample_time:.2f}s")

        if (i + 1) % 10 == 0:
            avg_time = total_time / (i + 1)
            remaining = (len(texts) - i - 1) * avg_time
            print(f"   Progress: {i+1}/{len(texts)} | Average: {avg_time:.2f}s/sample | Estimated remaining: {remaining/60:.1f}min")
        
        rows.append({
            'ID': sample_id,
            'Requirement': text,
            'PlantUML': plantuml_code,
            'Label': 'MATCH',
            'Error_Type': 'none',
            'Image_Path': img_path,
            'Dataset': 'dataset_A_PERFECT'
        })
    
    df = pd.DataFrame(rows)
    output_file = os.path.join(OUTPUT_DIR, "dataset_A_PERFECT.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\nDataset A completed")
    print(f"   File: {output_file}")
    print(f"   Samples: {len(df)}")
    print(f"   Total time: {total_time/60:.2f}min")

    return df

def generate_dataset_b(texts):
    print(f"\n{'='*40} Generating Dataset B: Synonym {'='*40}")
    print(f"Using first {len(texts)} rows of data")
    print(f"Label: MATCH (synonym change at requirement level, still matches)")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    rows = []
    total_time = 0
    
    for i, text in enumerate(texts):
        sample_id = f"dataset_B_SYNONYM_{i+1:04d}"
        debug_print = (i < DEBUG_PRINT_COUNT)

        if debug_print:
            print(f"\n{'='*25} Sample {i+1}/{len(texts)} - {sample_id} {'='*25}")
            print(f"Original Requirement: {text}")
        
        sample_start = time.time()
        
        # Step 1: Apply synonym change to requirement text
        changed_text, api_time1 = change_requirement_synonym(text, debug_print=debug_print)
        
        # Step 2: Generate PlantUML from changed requirement
        plantuml_code, api_time2 = generate_base_plantuml(changed_text, debug_print=debug_print)
        
        # Step 3: Generate image
        img_path = generate_image(plantuml_code, f"{sample_id}.png", error_type="synonym", requirement_original=text, requirement_changed=changed_text, image_dir=IMAGE_DIR_B)
        sample_time = time.time() - sample_start
        total_time += sample_time

        if debug_print:
            print(f"Sample total time: {sample_time:.2f}s (change: {api_time1:.2f}s + generation: {api_time2:.2f}s)")

        if (i + 1) % 10 == 0:
            avg_time = total_time / (i + 1)
            remaining = (len(texts) - i - 1) * avg_time
            print(f"   Progress: {i+1}/{len(texts)} | Average: {avg_time:.2f}s/sample | Estimated remaining: {remaining/60:.1f}min")
        
        rows.append({
            'ID': sample_id,
            'Requirement': text,
            'Requirement_Changed': changed_text,
            'PlantUML': plantuml_code,
            'Label': 'MATCH',
            'Error_Type': 'synonym',
            'Image_Path': img_path,
            'Dataset': 'dataset_B_SYNONYM'
        })
    
    df = pd.DataFrame(rows)
    output_file = os.path.join(OUTPUT_DIR, "dataset_B_SYNONYM.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\nDataset B completed")
    print(f"   File: {output_file}")
    print(f"   Samples: {len(df)}")
    print(f"   Total time: {total_time/60:.2f}min")

    return df

def generate_dataset_c(texts):
    print(f"\n{'='*40} Generating Dataset C: Logic Error {'='*40}")
    print(f"Using first {len(texts)} rows of data")
    print(f"Label: MISMATCH (logic error at requirement level)")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    rows = []
    total_time = 0
    
    for i, text in enumerate(texts):
        sample_id = f"dataset_C_LOGIC_ERROR_{i+1:04d}"
        debug_print = (i < DEBUG_PRINT_COUNT)

        if debug_print:
            print(f"\n{'='*25} Sample {i+1}/{len(texts)} - {sample_id} {'='*25}")
            print(f"Original Requirement: {text}")
        
        sample_start = time.time()
        
        # Step 1: Apply logic error change to requirement text
        changed_text, api_time1 = change_requirement_logic_error(text, debug_print=debug_print)
        
        # Step 2: Generate PlantUML from changed requirement (with logic error)
        plantuml_code, api_time2 = generate_base_plantuml(changed_text, debug_print=debug_print)
        
        # Step 3: Generate image
        img_path = generate_image(plantuml_code, f"{sample_id}.png", error_type="logic_error", requirement_original=text, requirement_changed=changed_text, image_dir=IMAGE_DIR_C)
        sample_time = time.time() - sample_start
        total_time += sample_time

        if debug_print:
            print(f"Sample total time: {sample_time:.2f}s (change: {api_time1:.2f}s + generation: {api_time2:.2f}s)")

        if (i + 1) % 10 == 0:
            avg_time = total_time / (i + 1)
            remaining = (len(texts) - i - 1) * avg_time
            print(f"   Progress: {i+1}/{len(texts)} | Average: {avg_time:.2f}s/sample | Estimated remaining: {remaining/60:.1f}min")
        
        rows.append({
            'ID': sample_id,
            'Requirement': text,
            'Requirement_Changed': changed_text,
            'PlantUML': plantuml_code,
            'Label': 'MISMATCH',
            'Error_Type': 'logic_error',
            'Image_Path': img_path,
            'Dataset': 'dataset_C_LOGIC_ERROR'
        })
    
    df = pd.DataFrame(rows)
    output_file = os.path.join(OUTPUT_DIR, "dataset_C_LOGIC_ERROR.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\nDataset C completed")
    print(f"   File: {output_file}")
    print(f"   Samples: {len(df)}")
    print(f"   Total time: {total_time/60:.2f}min")

    return df


if __name__ == "__main__":

    print("Dataset Generator")
    print(f"Model: {OPENAI_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Samples: {NUM_SAMPLES} per dataset")
    print(f"Debug: Show detailed info for first {DEBUG_PRINT_COUNT} samples")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load Kaggle Data
    print("\nLoading Kaggle data...")
    try:
        df_source = pd.read_csv(INPUT_CSV)

        raw_series = df_source["requirement"].dropna().astype(str)
        clean_series = raw_series.str.strip()

        unique_series = clean_series.drop_duplicates()
        all_texts = unique_series.tolist()

        print(f"   Loaded {len(raw_series)} rows from CSV")
        print(f"   Removed {len(raw_series) - len(all_texts)} duplicates; {len(all_texts)} unique requirements remaining")

        # Set random seed for reproducibility
        RANDOM_SEED = 40
        random.seed(RANDOM_SEED)
        print(f"   Random seed set to {RANDOM_SEED} for reproducibility")

        # Randomly select NUM_SAMPLES rows
        if len(all_texts) <= NUM_SAMPLES:
            texts = all_texts[:NUM_SAMPLES]
            print(f"   Only {len(all_texts)} available; using all of them")
        else:
            texts = random.sample(all_texts, NUM_SAMPLES)
            print(f"   Randomly selected {len(texts)} rows for processing")

    except Exception as e:
        print(f"   Loading failed: {e}")
        exit(1)
    
    program_start = time.time()

    # Generate three datasets
    df_a = generate_dataset_a(texts)
    df_b = generate_dataset_b(texts)
    df_c = generate_dataset_c(texts)

    program_time = time.time() - program_start

    # Summary
    print(f"\n{'='*20} Generation Complete - Statistics {'='*20}")
    print(f"Dataset A (Perfect Match): {len(df_a)} samples - MATCH")
    print(f"Dataset B (Synonym):       {len(df_b)} samples - MATCH")
    print(f"Dataset C (Logic Error):   {len(df_c)} samples - MISMATCH")
    print(f"Total time: {program_time/60:.2f} minutes")
    print(f"Average: {program_time/(len(df_a)+len(df_b)+len(df_c)):.2f} seconds/sample")
    print("API Usage Summary:")
    print(f"  Input tokens:   {total_input_tokens:,}")
    print(f"  Output tokens:  {total_output_tokens:,}")
    print(f"  Total tokens:   {total_input_tokens + total_output_tokens:,}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_DIR}/dataset_A_PERFECT.csv")
    print(f"  - {OUTPUT_DIR}/dataset_B_SYNONYM.csv")
    print(f"  - {OUTPUT_DIR}/dataset_C_LOGIC_ERROR.csv")
    print(f"  - {IMAGE_DIR}/ (images)")