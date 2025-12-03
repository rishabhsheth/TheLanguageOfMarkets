import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / 1024**2  # MB
    print(f"Total VRAM: {total_vram:.0f} MB")
    
    # Rough estimate: 50 MB per text of max_length=512
    mem_per_sample = 50  
    safe_vram = total_vram * 0.7  # 70% safety margin
    batch_size = int(safe_vram / mem_per_sample)
    batch_size = max(1, batch_size)  # ensure at least 1
    print(f"Recommended batch size: {batch_size}")
