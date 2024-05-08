import pandas as pd
from lorem_text import lorem
import os

def generate_messages(df: pd.DataFrame, message_dir: str):
    # Drop na for now
    df = df.dropna(subset=['modifiedPictureNrOfColourChannels', 'modifiedPictureNrOfPixels'])

    # Iterate over combinations and for every combination create random text file
    message_list = df.loc[(df['message'] != '-') & (df['message'] != 'Watermark.txt'), 'message'].unique().tolist()
    for message in message_list:
        er, r, c = message[:-4].split('_')
        if er == "1,00E-05":
            er = "000001"
        er = float(er[:1] + '.' + er[1:])
        r = int(r)
        c = int(c)

        # Calculate required number of bytes for this combination
        required_nr_of_bits = int(r * c * er)

        # Generate random text of length required_nr_of_bytes
        filename = f"{message_dir}/{str(er).replace('.', '')}_{str(r)}_{str(c)}.txt"
        if not os.path.exists(filename):
            print(f'{filename} does not exist')
            with open(filename, 'wb') as file:
                bits_written = 0
                while bits_written < required_nr_of_bits:
                    # Generate new message part
                    msg = lorem.words(1) + ' '
                    msg_bytes = bytes(msg, 'utf-8')

                    # If msg_bytes is greater than what is still necessary, remove superfluous bytes (as far as possible)
                    superfluous_bits = len(msg_bytes) * 8 + bits_written - required_nr_of_bits
                    if superfluous_bits > 0:
                        nr_of_chars_to_remove = round(superfluous_bits/8)
                        msg = msg[:-nr_of_chars_to_remove]
                        msg_bytes = bytes(msg, 'utf-8')
                        file.write(msg_bytes)
                        bits_written += len(msg_bytes) * 8  # 8 bits in a byte since only using characters of 8 bits in UTF-8
                        break

                    # Write bytes to file
                    file.write(msg_bytes)
                    bits_written += len(msg_bytes) * 8 # 8 bits in a byte since only using characters of 8 bits in UTF-8