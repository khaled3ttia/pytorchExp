

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #2
,addB%
#
	full_text

%7 = add i64 %6, 1
"i64B

	full_text


i64 %6
4truncB+
)
	full_text

%8 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
1addB*
(
	full_text

%9 = add nsw i32 %2, -1
4icmpB,
*
	full_text

%10 = icmp sgt i32 %9, %8
"i32B

	full_text


i32 %9
"i32B

	full_text


i32 %8
8brB2
0
	full_text#
!
br i1 %10, label %11, label %40
!i1B

	full_text


i1 %10
Ncall8BD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 1) #2
0add8B'
%
	full_text

%13 = add i64 %12, 1
%i648B

	full_text
	
i64 %12
8trunc8B-
+
	full_text

%14 = trunc i64 %13 to i32
%i648B

	full_text
	
i64 %13
5mul8B,
*
	full_text

%15 = mul nsw i32 %14, %2
%i328B

	full_text
	
i32 %14
1add8B(
&
	full_text

%16 = add i32 %15, %8
%i328B

	full_text
	
i32 %15
$i328B

	full_text


i32 %8
1mul8B(
&
	full_text

%17 = mul i32 %16, %1
%i328B

	full_text
	
i32 %16
5add8B,
*
	full_text

%18 = add nsw i32 %17, %1
%i328B

	full_text
	
i32 %17
0add8B'
%
	full_text

%19 = add i32 %4, -2
2add8B)
'
	full_text

%20 = add i32 %19, %18
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %18
6sext8B,
*
	full_text

%21 = sext i32 %20 to i64
%i328B

	full_text
	
i32 %20
^getelementptr8BK
I
	full_text<
:
8%22 = getelementptr inbounds double, double* %0, i64 %21
%i648B

	full_text
	
i64 %21
Abitcast8B4
2
	full_text%
#
!%23 = bitcast double* %22 to i64*
-double*8B

	full_text

double* %22
Hload8B>
<
	full_text/
-
+%24 = load i64, i64* %23, align 8, !tbaa !8
'i64*8B

	full_text


i64* %23
5add8B,
*
	full_text

%25 = add nsw i32 %17, %4
%i328B

	full_text
	
i32 %17
6sext8B,
*
	full_text

%26 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
^getelementptr8BK
I
	full_text<
:
8%27 = getelementptr inbounds double, double* %0, i64 %26
%i648B

	full_text
	
i64 %26
Abitcast8B4
2
	full_text%
#
!%28 = bitcast double* %27 to i64*
-double*8B

	full_text

double* %27
Hstore8B=
;
	full_text.
,
*store i64 %24, i64* %28, align 8, !tbaa !8
%i648B

	full_text
	
i64 %24
'i64*8B

	full_text


i64* %28
/add8B&
$
	full_text

%29 = add i32 %4, 1
2add8B)
'
	full_text

%30 = add i32 %29, %17
%i328B

	full_text
	
i32 %29
%i328B

	full_text
	
i32 %17
6sext8B,
*
	full_text

%31 = sext i32 %30 to i64
%i328B

	full_text
	
i32 %30
^getelementptr8BK
I
	full_text<
:
8%32 = getelementptr inbounds double, double* %0, i64 %31
%i648B

	full_text
	
i64 %31
Abitcast8B4
2
	full_text%
#
!%33 = bitcast double* %32 to i64*
-double*8B

	full_text

double* %32
Hload8B>
<
	full_text/
-
+%34 = load i64, i64* %33, align 8, !tbaa !8
'i64*8B

	full_text


i64* %33
0add8B'
%
	full_text

%35 = add i32 %4, -1
2add8B)
'
	full_text

%36 = add i32 %35, %18
%i328B

	full_text
	
i32 %35
%i328B

	full_text
	
i32 %18
6sext8B,
*
	full_text

%37 = sext i32 %36 to i64
%i328B

	full_text
	
i32 %36
^getelementptr8BK
I
	full_text<
:
8%38 = getelementptr inbounds double, double* %0, i64 %37
%i648B

	full_text
	
i64 %37
Abitcast8B4
2
	full_text%
#
!%39 = bitcast double* %38 to i64*
-double*8B

	full_text

double* %38
Hstore8B=
;
	full_text.
,
*store i64 %34, i64* %39, align 8, !tbaa !8
%i648B

	full_text
	
i64 %34
'i64*8B

	full_text


i64* %39
'br8B

	full_text

br label %40
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %1
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %2
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 1
$i328B

	full_text


i32 -2       	  
 
                     !    "# "" $% $$ &' && () (( *+ ** ,- ,, ./ .0 .. 11 23 24 22 56 55 78 77 9: 99 ;< ;; == >? >@ >> AB AA CD CC EF EE GH GI GG JL  L *L 7L CM M N N &N 1N =O O     	            !  #" % '& )( +* -$ /, 01 3 42 65 87 :9 <= ? @> BA DC F; HE I
 
 KJ K PP K PP  PP Q R R =S S 1T T U "
kernel_comm3_1"
_Z13get_global_idj*?
npb-MG-kernel_comm3_1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?
 
transfer_bytes_log1p
,??A

devmap_label
 

transfer_bytes
???

wgsize_log1p
,??A

wgsize
 