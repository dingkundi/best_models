while 1==1:
    BIM=float(input('请输入身高'))

    if BIM<18.5:
        print('过轻')
    elif BIM>=18.5 and BIM<=25:
        print('正常')
    elif BIM>=25 and BIM<=28:
        print('过重')
    else :
        print('过度严重肥胖')