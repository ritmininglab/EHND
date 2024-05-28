if [ "$1" = "imagenet" ] && [ "$2" = "relabel" ]; then
    python train.py -gpu -data ImageNet -test -m RLB -bsize 0 -nep 1000 -nlrd 20 -nl 1 -rl 20 -lrd 0.7
elif [ "$1" = "imagenet" ] && [ "$2" = "ehnd" ]; then
    python train.py -gpu -data ImageNet -test -m EHND -bsize 0 -nep 5000 -nlrd 20 -ehnd 1 -cw -lrd 0.7
elif [ "$1" = "imagenet" ] && [ "$2" = "td" ]; then
    python train.py -gpu -data ImageNet -test -m TD -bsize 0 -nep 1200 -nlrd 2 -ex 1
elif [ "$1" = "imagenet" ] && [ "$2" = "td+ehnd" ]; then
    python train.py -gpu -data ImageNet -test -m TD+EHND -bsize 0 -nep 1000 -nlrd 10 -tdname "TD_-1_1e+00_0e+00_1e-02_1e-02" -ehnd 1 -cw -relu -sm l -lrd 0.7
elif [ "$1" = "awa2" ] && [ "$2" = "ehnd" ]; then
    python train.py -gpu -data AWA2 -test -m EHND -bsize 0 -nep 1000 -nlrd 0 -ehnd 1
elif [ "$1" = "awa2" ] && [ "$2" = "td" ]; then
    python train.py -gpu -data AWA2 -test -m TD -bsize 0 -nep 1000 -nlrd 0 -ex 1 -cw
elif [ "$1" = "awa2" ] && [ "$2" = "td+ehnd" ]; then
    python train.py -gpu -data AWA2 -test -m TD+EHND -bsize 0 -nep 1000 -nlrd 0 -tdname "TD_-1_1e+00_0e+00_cw_1e-02_1e-02" -ehnd 1 -sm n
elif [ "$1" = "cub" ] && [ "$2" = "ehnd" ]; then
    python train.py -gpu -data CUB -test -m EHND -bsize 0 -nep 5000 -ehnd 1 -lrd 0.7 -nlrd 20
    # python train.py -gpu -data CUB -test -m EHND -bsize 0 -nep 10000 -nlrd 20   -lrd 0.7 -ehnd 1
elif [ "$1" = "cub" ] && [ "$2" = "td" ]; then
    python train.py -gpu -data CUB -test -m TD -bsize 0 -nep 1000 -nlrd 0 -ex 1 -cw
elif [ "$1" = "cub" ] && [ "$2" = "td+ehnd" ]; then
    python train.py -gpu -data CUB -test -m TD+EHND -bsize 0 -nep 10000 -nlrd 20  -lrd 0.7 -tdname "TD_-1_1e+00_0e+00_cw_1e-02_1e-02" -ehnd 1 -relu -sm l 
else
    echo "Usage: sh train.sh {d} {m}"
    echo "{d} = imagenet, awa2, cub"
    echo "{m} = ehnd, td, td+ehnd"
fi