import argparse
import datetime
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../ner-data', help='Directory containing the data')
    parser.add_argument('--save_dir', type=str, default='../output', help='Directory containing the data')
    parser.add_argument('--output_file', type=str, default='./output.txt', help='Directory containing the data')
    parser.add_argument('--checkpoint', type=str, default=False, help='Directory containing the data')
    parser.add_argument('--checkpoint_path', type=str, default='./2024-05-30 01:32:01/model_weights_18.pth', help='Directory containing the data')
    parser.add_argument('--local_files_only', type=bool, default=True, help='Directory containing the data')
    parser.add_argument('--train_length', type=int, default=300, help='Directory containing the data')
    
    parser.add_argument('--max_length', type=int, default=2048, help='Directory containing the data')
    parser.add_argument('--d_model', type=int, default=512, help='Directory containing the data')
    parser.add_argument('--num_layers', type=int, default=6, help='Directory containing the data')
    parser.add_argument('--num_heads', type=int, default=8, help='Directory containing the data')
    parser.add_argument('--ffn_hidden', type=int, default=1024, help='Directory containing the data')
    parser.add_argument('--dropout', type=float, default=0.1, help='Directory containing the data')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Directory containing the data')
    parser.add_argument('--batch_size', type=int, default=64, help='Directory containing the data')
    parser.add_argument('--num_workers', type=int, default=4, help='Directory containing the data')
    parser.add_argument('--epochs', type=int, default=20, help='Directory containing the data')
    parser.add_argument('--device', type=str, default='cuda', help='Directory containing the data')
    parser.add_argument('--project_name', type=str, default='transformer', help='Directory containing the data')
    parser.add_argument('--log_ct', type=int, default=100, help='Directory containing the data')
    # 继续添加其他参数...
    return parser.parse_args()
