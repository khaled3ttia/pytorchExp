

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #2
,addB%
#
	full_text

%6 = add i64 %5, 1
"i64B

	full_text


i64 %5
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
3icmpB+
)
	full_text

%8 = icmp sgt i32 %7, %2
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %21, label %9
 i1B

	full_text	

i1 %8
5sext8B+
)
	full_text

%10 = sext i32 %3 to i64
Xgetelementptr8BE
C
	full_text6
4
2%11 = getelementptr inbounds i32, i32* %1, i64 %10
%i648B

	full_text
	
i64 %10
0shl8B'
%
	full_text

%12 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%13 = ashr exact i64 %12, 32
%i648B

	full_text
	
i64 %12
Xgetelementptr8BE
C
	full_text6
4
2%14 = getelementptr inbounds i32, i32* %0, i64 %13
%i648B

	full_text
	
i64 %13
Hload8B>
<
	full_text/
-
+%15 = load i32, i32* %14, align 4, !tbaa !8
'i32*8B

	full_text


i32* %14
:add8B1
/
	full_text"
 
%16 = add i64 %12, -4294967296
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%17 = ashr exact i64 %16, 32
%i648B

	full_text
	
i64 %16
Ygetelementptr8BF
D
	full_text7
5
3%18 = getelementptr inbounds i32, i32* %11, i64 %17
'i32*8B

	full_text


i32* %11
%i648B

	full_text
	
i64 %17
Hload8B>
<
	full_text/
-
+%19 = load i32, i32* %18, align 4, !tbaa !8
'i32*8B

	full_text


i32* %18
6sub8B-
+
	full_text

%20 = sub nsw i32 %15, %19
%i328B

	full_text
	
i32 %15
%i328B

	full_text
	
i32 %19
Hstore8B=
;
	full_text.
,
*store i32 %20, i32* %14, align 4, !tbaa !8
%i328B

	full_text
	
i32 %20
'i32*8B

	full_text


i32* %14
'br8B

	full_text

br label %21
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %2
&i32*8B

	full_text
	
i32* %0
&i32*8B

	full_text
	
i32* %1
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
-i648B"
 
	full_text

i64 -4294967296
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 1       	 
                        !" !# !! $& 
' ( )     	
              " # % 
$ % ** % ** + , - - - . "	
makea_7"
_Z13get_global_idj*?
npb-CG-makea_7.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize
?
 
transfer_bytes_log1p
N?jA

wgsize_log1p
N?jA

devmap_label
 

transfer_bytes
???