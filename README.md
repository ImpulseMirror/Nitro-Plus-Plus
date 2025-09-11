# Nitro Plus Plus
Technology has caught up with the lazy. This is an experiment in what is possible when you choose not the "support the official translation"


## Poetry Commands
## Install dependencies and build wheel
```
poetry lock
poetry install -with cuda
```
### Start request server
```
poetry run nitropp serve --model org/repo
```
### Kill server 

```
poetry run killport 8787 # kills the process on port 8787
```

# Translation Endpoints

## Example Request

```
curl --location 'http://127.0.0.1:8787/translate' \
--header 'Content-Type: application/json; charset=utf-8' \
--data '{"text":"双子の兄達は3歳で魔力が発現しました。"}'
```
## Response

```
{
    "source": "双子の兄達は3歳で魔力が発現しました。",
    "translation": "The twin boys both developed magic powers at the age of three."
}
```


## Example Batch Request
```
curl --location 'http://127.0.0.1:8787/batch_translate' \
--header 'Content-Type: application/json; charset=utf-8' \
--data '{"texts":["日本語のテキスト","空気をよめなさい","秋雨前線の影響などで、広い範囲で大気の状態が不安定になり東海などでは局地的に非常に激しい雨が降っています。"],"max_new":128,"temp":0.0,"outfile":"outputs/weather_batch.json"}'
```

## Example Response
```
{
    "count": 3,
    "outfile": "E:\\Source\\Nitro-Plus-Plus\\outputs\\weather_batch.json",
    "results": [
        {
            "index": 1,
            "source": "日本語のテキスト",
            "translation": "Japanese Text",
            "notes": ""
        },
        {
            "index": 2,
            "source": "空気をよめなさい",
            "translation": "Read the air",
            "notes": ""
        },
        {
            "index": 3,
            "source": "秋雨前線の影響などで、広い範囲で大気の状態が不安定になり東海などでは局地的に非常に激しい雨が降っています。",
            "translation": "Due to the influence of the autumn rain front, the atmospheric conditions have become unstable across a wide area, and intense rainfall is occurring in local areas, including in the Toukai region.",
            "notes": ""
        }
    ]
}
```